
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

    my_int: IntProperty(
        name = "Int Value",
        description="A integer property",
        default = 23,
        min = 10,
        max = 100
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
        

class OBJECT_OT_automask(Operator):
    bl_idname = "object.automask"
    bl_label = "AutoMask Operator"
    bl_description = "Track the selected mask \nto the next frame"

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

    def execute(self, context):
        mask = context.space_data.mask
        maskSpline = mask.layers.active.splines[0]
        points = maskSpline.points
        maskSpline.use_cyclic = True
        clip = context.space_data.clip
        self.hw = clip.size
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
        framenum = bpy.context.scene.frame_current
        movpath = bpy.path.abspath(clip.filepath)
        co, lhand, rhand = [], [], []
        for p in points:
            # need types to be free as it is the most general type
            p.handle_left_type = 'FREE'
            p.handle_right_type = 'FREE'
            co.append(self.absolute_coord(p.co))
            lhand.append(self.absolute_coord(p.handle_left))
            rhand.append(self.absolute_coord(p.handle_right))
        # collection of coordinates and handles
        crl = [co, rhand, lhand]
        # get mask from the point coordinates
        mask = crl2mask(crl, int(self.hw[0]), int(self.hw[1]))

        # load model
        try:
            state = bpy.state
            model = bpy.model
        except AttributeError:
            # must first initialize model
            p = context.scene.settings.automask_path
            if p == '':
                raise ValueError('AutoMask path is empty.')
            state = p
            model = None
        next_mask, state, model = track_object(model, state, mask, movpath, framenum)
        if type(next_mask) == dict:
            return next_mask
        success, crl = fit2mask(next_mask)

        success = success and state['score'] > .8

        if not success:
            return {'FINISHED'}
        co, rh, lh = crl
        print(co)
        print([len(l) for l in crl])
        bpy.context.scene.frame_set(framenum+1)
        # bpy.ops.clip.keyframe_insert()
        # create more points in the mask if needed
        N, newN = len(points), len(co)
        print(N, newN)

        if newN > N:
            points.add(newN-N)
            for i in range(1, newN-N+1):
                self.copy_point_attributes(points[0], points[-i])
        elif newN < N:
            for i in range(1, N-newN+1):
                points.remove(points[-1])

        # change handles to the found optimum position
        for i, p in enumerate(points):
            p.co.x, p.co.y = self.relative_coord(co[i])
            p.handle_left.x, p.handle_left.y = self.relative_coord(lh[i])
            p.handle_right.x, p.handle_right.y = self.relative_coord(rh[i])
        bpy.state = state
        bpy.model = model

        return {'FINISHED'}


class PANEL0_PT_automask(Panel):
    bl_label = "AutoMask"
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
        row = layout.row()
        row.operator("object.automask", text="Mask")
        row = layout.column()
        layout.prop(settings, 'automask_path')

        layout.separator()


classes = (OBJECT_OT_automask, PANEL0_PT_automask, Settings)


def register():

    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.settings = PointerProperty(type=Settings)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    try:
        del bpy.state
        del bpy.model
    except AttributeError:
        pass
    del bpy.types.Scene.settings

if __name__ == "__main__":
    register()
