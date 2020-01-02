
from __future__ import print_function
import bpy
from bpy.types import Operator, Panel, PropertyGroup, WindowManager
from bpy.props import PointerProperty, StringProperty, IntProperty, FloatProperty, BoolProperty
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
        name="Directions",
        description="The lower this value is the more points will be created",
        default=3,
        min=1,
        max=5
    )

    maxlen: IntProperty(
        name="Max. Length",
        description="The maximum amount of pixels a mask line segment is tracing",
        default=150,
        min=1
    )

    threshold: IntProperty(
        name="Treshold",
        description="The amount of points that can point in a different direction\nbefore a new segment is created",
        default=10,
        min=0
    )

    my_float: FloatProperty(
        name="Float Value",
        description="A float property",
        default=23.7,
        min=0.01,
        max=30.0
    )

    change_layer: BoolProperty(
        name="Change Layer",
        description="Change the active Mask Layer according to the frame\nwhen moving the along the timeline",
        default=True
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

    def hide_layer(self, layer, hide=True):
        layer.hide = hide
        layer.hide_render = hide
        layer.hide_select = hide
        layer.keyframe_insert('hide')
        layer.keyframe_insert('hide_render')
        layer.keyframe_insert('hide_select')

    def automask(self, context, model, state, movpath):
        mask = context.space_data.mask
        settings = context.scene.settings
        layer = mask.layers.active
        maskSplines = layer.splines
        co_tot, lhand_tot, rhand_tot = [], [], []
        framenum = bpy.context.scene.frame_current
        try:
            _=int(layer.name.split('.f')[-1])
        except ValueError:
            # no frame identification in the masklayer name
            layer.name = layer.name + '.f%i'%framenum
        for i, maskSpline in enumerate(maskSplines):
            points = maskSpline.points
            maskSpline.use_cyclic = True
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
            curr_mask = crl2mask(crl, int(self.hw[0]), int(self.hw[1]))

            # load model
            next_mask, state, model = track_object(model, state, curr_mask, movpath, framenum)
            if next_mask is None:
                return {'CANCELLED'}
            # trace mask returned by SiamMask
            success, crl = fit2mask(next_mask, maxnum=settings.maxnum, threshold=settings.threshold, maxlen=settings.maxlen)
            success = success and state['score'] > .8
            if not success:
                return {'CANCELLED'}
            # save handle positions for each spline, so we can change the position later
            co, rh, lh = crl
            co_tot.append(co)
            rhand_tot.append(rh)
            lhand_tot.append(lh)
        #propagate in time
        bpy.ops.ed.undo_push()
        self.hide_layer(layer, False)
        name = '%s.f%i' % (layer.name.split('.f')[0], framenum+1)
        new_layer = mask.layers.get(name)
        if new_layer is None:
            new_layer = mask.layers.new(name=name)
        else:
            for spline in new_layer.splines:
                new_layer.splines.remove(spline)
        self.hide_layer(new_layer, True)
        mask.layers.active = new_layer
        bpy.ops.clip.change_frame(frame=framenum+1)
        self.hide_layer(layer, True)
        self.hide_layer(new_layer, False)
        for i in range(len(maskSplines)):
            maskSpline = new_layer.splines.new()
            points = maskSpline.points
            maskSpline.use_cyclic = True
            co, rh, lh = co_tot[i], rhand_tot[i], lhand_tot[i]
            # create points in the mask if needed
            N, newN = len(points), len(co)
            if newN > N:
                points.add(newN-N)
                for i in range(1, newN-N+1):
                    self.copy_point_attributes(p, points[-i])

            # change handles to the found optimum position
            for i, p in enumerate(points):
                p.co.x, p.co.y = self.relative_coord(co[i])
                p.handle_left.x, p.handle_left.y = self.relative_coord(lh[i])
                p.handle_right.x, p.handle_right.y = self.relative_coord(rh[i])
        return model, state


class OBJECT_OT_automask_single(Operator):
    bl_idname = "object.automask_single"
    bl_label = ""
    bl_description = "Track the selected mask \nto the next frame"

    def execute(self, context):
        clip = context.space_data.clip
        movpath = bpy.path.abspath(clip.filepath)
        amh = AutoMask_helper()
        amh.hw = clip.size
        amh.set_coordinate_transform()
        proj_dir = paths[-1]
        if proj_dir == '':
            raise ValueError('AutoMask path is empty.')
        state = proj_dir  # set first state to proj_dir
        model = None
        ret = amh.automask(context, model, state, movpath)
        if type(ret) == dict:
            return ret
        del ret
        return {'FINISHED'}


class OBJECT_OT_automask(Operator):
    bl_idname = "object.automask"
    bl_label = ""
    bl_description = "Track the selected mask\nunitl it is lost"

    _updating = False
    _calcs_done = True
    _timer = None

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._calcs_done = True
        elif event.type == 'TIMER' and not self._updating and not self._calcs_done:
            self._updating = True
            frame_end = context.scene.frame_end
            if bpy.context.scene.frame_current < frame_end-1:
                ret = self.amh.automask(context, self.model, self.state, self.amh.movpath)
                if type(ret) == dict:
                    self._calcs_done = True
                else:
                    self.model = ret[0]
                    self.state = ret[1]
            self._updating = False
        if self._calcs_done:
            self.cancel(context)

        return {'PASS_THROUGH'}

    def execute(self, context):
        clip = context.space_data.clip
        self.amh = AutoMask_helper()
        self.amh.movpath = bpy.path.abspath(clip.filepath)
        self.amh.hw = clip.size
        self.amh.set_coordinate_transform()
        proj_dir = paths[-1]
        if proj_dir == '':
            raise ValueError('AutoMask path is empty.')
        self.state = proj_dir  # set first state to proj_dir
        self.model = None
        self._calcs_done = False
        context.window_manager.modal_handler_add(self)
        self._updating = False
        self._timer = context.window_manager.event_timer_add(.05, window=context.window)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            del self.model
            del self.state
        return {'CANCELLED'}


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
        layout.prop(settings, 'threshold')
        layout.prop(settings, 'maxnum')
        layout.prop(settings, 'change_layer')

        layout.separator()

def MaskLayerActivater(scene):
    if scene.settings.change_layer:
        f = scene.frame_current
        masks=bpy.data.masks
        for m in masks:
            layers=m.layers 
            for l in layers:
                try:
                    l_num=int(l.name.split('.f')[-1])
                except ValueError:
                    continue
                if f==l_num:
                    layers.active=l
                    break

classes = (OBJECT_OT_automask_single, OBJECT_OT_automask, PANEL0_PT_automask, Settings)
addon_keymaps = []

def remove_handler():
    my_handler_list = bpy.app.handlers.frame_change_pre
    fin = len(my_handler_list)
    for idx, func in enumerate(reversed(my_handler_list)):
        if func.__name__ == 'MaskLayerActivater':
            my_handler_list.pop(fin-1-idx)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.settings = PointerProperty(type=Settings)
    remove_handler()
    bpy.app.handlers.frame_change_pre.append(MaskLayerActivater)   

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.settings
    remove_handler()

if __name__ == "__main__":
    register()
