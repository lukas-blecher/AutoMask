
from __future__ import print_function
import bpy
from bpy.types import Operator, Panel, PropertyGroup, WindowManager
from bpy.props import PointerProperty, StringProperty
import os


bl_info = {
    'name': 'AutoMask',
    'category': 'Motion Tracking',
    'location': 'Movie clip Editor > Tools Panel > AutoMask',
}

class AutoMask(Operator):
    bl_idname = "object.automask"
    bl_label = "AutoMask Operator"
    
    def big_trans(self,inv=False):
        return lambda x:x
    
    def small_trans(self,inv=False):
        frac=max(self.hw)/float(min(self.hw))
        off=.5-1/(2.*frac)
        if not inv:
            return lambda x:(x-off)*frac
        else:
            return lambda x:(x/frac)+off
    
    def absolute_coord(self, coordinate):
        width,height=self.hw
        coord = coordinate.copy()
        return [self.xtrans(coord.x)*width, (1-self.ytrans(coord.y))*height]
    
    def relative_coord(self,coordinate):
        width,height=self.hw
        return [self.xinvt(coordinate[0]/float(width)),self.yinvt(1-(coordinate[1]/float(height)))]
    
    def execute(self, context):
        mask=context.space_data.mask
        points=mask.layers.active.splines[0].points
        clip=context.space_data.clip
        self.hw=clip.size
        if self.hw[0]<self.hw[1]:
            self.xtrans=self.small_trans()
            self.xinvt=self.small_trans(True)
            self.ytrans=self.big_trans()
            self.yinvt=self.big_trans()
        elif self.hw[0]>self.hw[1]:
            self.ytrans=self.small_trans()
            self.yinvt=self.small_trans(True)
            self.xtrans=self.big_trans()
            self.xinvt=self.big_trans()
        else:
            self.xtrans=self.big_trans()
            self.xinvt=self.big_trans()
            self.ytrans=self.big_trans()
            self.yinvt=self.big_trans()
        framenum=bpy.context.scene.frame_current
        movpath=bpy.path.abspath(clip.filepath)
        co,lhand,rhand=[],[],[]
        for p in points:
            # need types to be free
            p.handle_left_type='FREE'
            p.handle_right_type='FREE'
            co.append(self.absolute_coord(p.co))
            lhand.append(self.absolute_coord(p.handle_left))
            rhand.append(self.absolute_coord(p.handle_right))
            

        return {'FINISHED'}


class AMPanel(Panel):
    bl_label = "AutoMask"
    bl_idname = "automask"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "AutoMask"

    @classmethod
    def poll(cls, context):
        return (context.area.spaces.active.clip is not None)

    # Draw UI
    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        row = layout.row()
        props = row.operator("object.automask", text="Mask")
        row = layout.column()
        row.prop(context.scene, 'out_path')

        layout.separator()


def register():
    # bpy.utils.register_class(TrackingExport)
    #WindowManager.autotracker_props = PointerProperty(type=TrackingExport)
    bpy.utils.register_module(__name__)
    bpy.types.Scene.out_path = StringProperty(
        name="Path",
        default="",
        description="Define the path to the AutoMask project directory",
        subtype='DIR_PATH'
    )
    # initialize neural network

def unregister():
    # bpy.utils.unregister_class(TrackingExport)
    bpy.utils.unregister_module(__name__)
    #del WindowManager.autotracker_props
    del bpy.types.Scene.conf_path


if __name__ == "__main__":
    register()
