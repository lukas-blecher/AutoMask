
from __future__ import print_function
import bpy
from bpy.types import Operator, Panel, PropertyGroup, WindowManager
from bpy.props import PointerProperty, StringProperty, IntProperty, FloatProperty
import sys
paths = [
    r'C:\Users\user\AppData\Local\Continuum\anaconda3\lib\site-packages',
    r'C:\Users\user\Desktop\ML\AutoMask'
]
for p in paths:
    sys.path.insert(0, p)
from mask_spline import *
from SiamMask import *

bl_info = {
    'blender': (2, 80, 0),
    'name': 'AutoMask',
    'category': 'Motion Tracking',
    'location': 'Masking> Movie Clip Editor > AutoMask',
    'author': 'Lukas Blecher'
}

class Settings(PropertyGroup):

    maxnum: IntProperty(
        name = "Directions",
        description="The lower this value is the more points will be created",
        default = 3,
        min = 1,
        max = 5
        )

    maxlen: IntProperty(
        name = "Max. Length",
        description="The maximum amount of pixels a mask line segment is tracing",
        default = 150,
        min = 1
        )

    my_float: FloatProperty(
        name = "Float Value",
        description = "A float property",
        default = 23.7,
        min = 0.01,
        max = 30.0
        )

    automask_path: StringProperty(
        name = "Project Directory",
        description="Choose a directory:",
        default=paths[-1],
        maxlen=1024,
        subtype='DIR_PATH'
        )
        
class AutoMask_helper:
    def big_trans(self, inv=False):
        return lambda x: x

    def small_trans(self, inv=False):
        frac = max(self.hw)/float(min(self.hw))
        off = .5-1/(2.*frac)
        if not inv:
            return lambda x: (x-off)*frac
        else:
            return lambda x: (x/frac)+off

    def copy_point_attributes(self, point, new_point):
        attributes = ['co', 'handle_left', 'handle_left_type', 'handle_right', 'handle_right_type', 'handle_type', 'weight']
        for attr in attributes:
            setattr(new_point, attr, getattr(point, attr))

    def absolute_coord(self, coordinate):
        width, height = self.hw
        coord = coordinate.copy()
        return [self.xtrans(coord.x)*width, (1-self.ytrans(coord.y))*height]

    def relative_coord(self, coordinate):
        width, height = self.hw
        return [self.xinvt(coordinate[0]/float(width)), self.yinvt(1-(coordinate[1]/float(height)))]

    def set_coordinate_transform(self):
        if self.hw[0] < self.hw[1]:
            self.xtrans = self.small_trans()
            self.xinvt = self.small_trans(True)
            self.ytrans = self.big_trans()
            self.yinvt = self.big_trans()
        elif self.hw[0] > self.hw[1]:
            self.ytrans = self.small_trans()
            self.yinvt = self.small_trans(True)
            self.xtrans = self.big_trans()
            self.xinvt = self.big_trans()
        else:
            self.xtrans = self.big_trans()
            self.xinvt = self.big_trans()
            self.ytrans = self.big_trans()
            self.yinvt = self.big_trans()

class OBJECT_OT_automask_single(Operator):
    bl_idname = "object.automask_single"
    bl_label = ""
    bl_description = "Track the selected mask \nto the next frame"

    def execute(self, context):
        mask = context.space_data.mask
        settings = context.scene.settings
        maskSpline = mask.layers.active.splines[0]
        points = maskSpline.points
        maskSpline.use_cyclic = True
        clip = context.space_data.clip
        amh=AutoMask_helper()
        amh.hw = clip.size
        amh.set_coordinate_transform()
        framenum = bpy.context.scene.frame_current
        movpath = bpy.path.abspath(clip.filepath)
        co, lhand, rhand = [], [], []
        for p in points:
            # need types to be free as it is the most general type
            p.handle_left_type = 'FREE'
            p.handle_right_type = 'FREE'
            co.append(amh.absolute_coord(p.co))
            lhand.append(amh.absolute_coord(p.handle_left))
            rhand.append(amh.absolute_coord(p.handle_right))
        # collection of coordinates and handles
        crl = [co, rhand, lhand]
        # get mask from the point coordinates
        mask = crl2mask(crl, int(amh.hw[0]), int(amh.hw[1]))

        # load model
        
        # must first initialize model
        p = context.scene.settings.automask_path
        if p == '':
            raise ValueError('AutoMask path is empty.')
        state = p
        model = None
        next_mask, state, model = track_object(model, state, mask, movpath, framenum)
        if type(next_mask) == dict:
            return next_mask
        success, crl = fit2mask(next_mask, maxnum=settings.maxnum, maxlen=settings.maxlen)

        success = success and state['score'] > .8

        if not success:
            return {'FINISHED'}
        co, rh, lh = crl

        #bpy.ops.clip.keyframe_insert()
        bpy.ops.ed.undo_push()
        bpy.ops.clip.change_frame(frame=framenum+1)

        # create more points in the mask if needed
        N, newN = len(points), len(co)
        if newN > N:
            points.add(newN-N)
            for i in range(1, newN-N+1):
                amh.copy_point_attributes(points[0], points[-i])
        elif newN < N:
            for i in range(1, N-newN+1):
                points.remove(points[-1])

        # change handles to the found optimum position
        for i, p in enumerate(points):
            p.co.x, p.co.y = amh.relative_coord(co[i])
            p.handle_left.x, p.handle_left.y = amh.relative_coord(lh[i])
            p.handle_right.x, p.handle_right.y = amh.relative_coord(rh[i])
        # insert keyframe
        #for p in points:
        #    p.keyframe_insert(data_path="co")

        return {'FINISHED'}

class OBJECT_OT_automask(Operator):
    bl_idname = "object.automask"
    bl_label = ""
    bl_description = "Track the selected mask \nuntil the object is lost"

    def execute(self, context):
        mask = context.space_data.mask
        settings = context.scene.settings
        maskSpline = mask.layers.active.splines[0]
        points = maskSpline.points
        maskSpline.use_cyclic = True
        clip = context.space_data.clip
        amh=AutoMask_helper()
        amh.hw = clip.size
        amh.set_coordinate_transform()
        framenum = bpy.context.scene.frame_current
        movpath = bpy.path.abspath(clip.filepath)
        co, lhand, rhand = [], [], []
        for p in points:
            # need types to be free as it is the most general type
            p.handle_left_type = 'FREE'
            p.handle_right_type = 'FREE'
            co.append(amh.absolute_coord(p.co))
            lhand.append(amh.absolute_coord(p.handle_left))
            rhand.append(amh.absolute_coord(p.handle_right))
        # collection of coordinates and handles
        crl = [co, rhand, lhand]

        # must first initialize model
        p = context.scene.settings.automask_path
        if p == '':
            raise ValueError('AutoMask path is empty.')
        state = p
        model = None
        success = True
        frame_end = context.scene.frame_end
        while success and framenum < frame_end:
            framenum=bpy.context.scene.frame_current
            # get mask from the point coordinates
            mask = crl2mask(crl, int(amh.hw[0]), int(amh.hw[1]))
            next_mask, state, model = track_object(model, state, mask, movpath, framenum)
            if type(next_mask) == dict:
                return next_mask
            success, crl = fit2mask(next_mask, maxnum=settings.maxnum, maxlen=settings.maxlen)

            success = success and state['score'] > .8

            if not success:
                return {'FINISHED'}
            co, rh, lh = crl

            #bpy.ops.clip.keyframe_insert()
            bpy.ops.ed.undo_push()
            bpy.ops.clip.change_frame(frame=framenum+1)

            # create more points in the mask if needed
            N, newN = len(points), len(co)
            if newN > N:
                points.add(newN-N)
                for i in range(1, newN-N+1):
                    amh.copy_point_attributes(points[0], points[-i])
            elif newN < N:
                for i in range(1, N-newN+1):
                    points.remove(points[-1])

            # change handles to the found optimum position
            for i, p in enumerate(points):
                p.co.x, p.co.y = amh.relative_coord(co[i])
                p.handle_left.x, p.handle_left.y = amh.relative_coord(lh[i])
                p.handle_right.x, p.handle_right.y = amh.relative_coord(rh[i])
            # insert keyframe
            #for p in points:
            #    p.keyframe_insert(data_path="co")

        return {'FINISHED'}

class PANEL0_PT_automask(Panel):
    bl_label = "Mask Tracking"
    bl_idname = "PANEL0_PT_automask"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoMask"

    @classmethod
    def poll(cls, context):
        return (context.area.spaces.active.clip is not None)

    # Draw UI
    def draw(self, context):
        settings = context.scene.settings
        layout = self.layout
        layout.use_property_split = True  # Active single-column layout
        c = layout.column()
        row = c.row()
        split = row.split(factor=0.3)
        c = split.column()
        c.label(text="Track:")
        split = split.split()
        c = split.row()
        c.operator("object.automask", icon="TRACKING_FORWARDS")
        c.operator("object.automask_single", icon="TRACKING_FORWARDS_SINGLE")
        row = layout.column()
        layout.prop(settings, 'maxlen')
        layout.prop(settings, 'maxnum')
        #layout.prop(settings, 'automask_path')

        layout.separator()


classes = (OBJECT_OT_automask_single, OBJECT_OT_automask, PANEL0_PT_automask, Settings)
addon_keymaps = []


def register():

    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.settings = PointerProperty(type=Settings)
    # handle the keymap
    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='automask', space_type='CLIP_EDITOR')
    kmi = km.keymap_items.new(OBJECT_OT_automask_single.bl_idname, 'RIGHT_ARROW', 'PRESS', alt=True)
    addon_keymaps.append(km)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.settings
    # handle the keymap
    wm = bpy.context.window_manager
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    # clear the list
    del addon_keymaps[:]

if __name__ == "__main__":
    register()
