import glfw
import torch
from imgui_bundle import imgui
from imgui_bundle import imgui_toggle

from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer
from easyvolcap.utils.data_utils import to_cuda, Visualization
from easyvolcap.utils.viewer_utils import Camera, CameraPath
from easyvolcap.utils.console_utils import *


class Viewer(VolumetricVideoViewer):
    def __init__(self,
                 window_size=[1080, 1920],  # height, width
                 window_title: str = f'EasyVolcap Viewer Custom Window',  # MARK: global config
                 fullscreen: bool = False,

                 camera_cfg: dotdict = dotdict(type=Camera.__name__, string='{"H":2032,"W":3840,"K":[[4279.6650390625,0.0,1920.0],[0.0,4279.6650390625,992.4420776367188],[0.0,0.0,1.0]],"R":[[0.41155678033828735,0.911384105682373,0.0],[-0.8666263818740845,0.39134538173675537,0.3095237910747528],[0.2820950746536255,-0.12738661468029022,0.9508903622627258]],"T":[[-4.033830642700195],[-1.7978200912475586],[3.9347341060638428]],"n":0.10000000149011612,"f":1000.0,"t":0.0,"v":0.0,"bounds":[[-10.0,-10.0,-3.0],[10.0,10.0,4.0]],"mass":0.10000000149011612,"moment_of_inertia":0.10000000149011612,"movement_force":1.0,"movement_torque":1.0,"movement_speed":5.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,0.0,-1.0]}'),

                 ):
        # Camera related configurations
        self.camera_cfg = camera_cfg
        self.fullscreen = fullscreen
        self.window_size = window_size
        self.window_title = window_title

        self.use_vsync = False
        self.font_size = 18
        self.font_bold: str = f'{dirname(__file__)}/../../assets/fonts/CascadiaCodePL-Bold.otf'
        self.font_italic: str = f'{dirname(__file__)}/../../assets/fonts/CascadiaCodePL-Italic.otf'
        self.font_default: str = f'{dirname(__file__)}/../../assets/fonts/CascadiaCodePL-Regular.otf'
        self.icon_file: str = f'{dirname(__file__)}/../../assets/imgs/easyvolcap.png'

        self.init_camera(camera_cfg)  # prepare for the actual rendering now, needs dataset -> needs runner
        self.init_glfw()  # ?: this will open up the window and let the user wait, should we move this up?
        self.init_imgui()

        from easyvolcap.engine import args
        args.type = 'gui'  # manually setting this parameter
        self.use_quad_cuda = True
        self.compose = False
        self.compose_power = 1.0
        self.init_opengl()
        self.init_quad()
        self.bind_callbacks()

        self.meshes = []

        self.camera_path = CameraPath()
        self.visualize_axes = True
        self.visualize_paths = True
        self.visualize_cameras = True
        self.visualize_bounds = True
        self.exposure = 1.0
        self.offset = 0.0
        self.epoch = 0
        self.dataset = dotdict(duration=1.0)
        self.visualization_type = Visualization.RENDER
        self.playing = False
        self.discrete_t = False
        self.playing_speed = 0.0
        self.network_available = False

        self.show_demo_window = False
        self.show_metrics_window = False
        self.skip_exception = False

        self.render_network = True
        self.render_meshes = True

        self.update_fps_time = 0.5
        self.update_mem_time = 0.5

        # Others
        self.static = dotdict(batch=dotdict(), output=dotdict())  # static data store updated through the rendering
        self.dynamic = dotdict()
        self.runner = dotdict(ep_iter=0, collect_timing=False, timer_record_to_file=False, timer_sync_cuda=True)

    def init_camera(self, camera_cfg: dotdict):
        self.camera = Camera(**camera_cfg)
        self.camera.front = self.camera.front  # perform alignment correction

    def frame(self):
        # print(f'framing: {time.perf_counter()}')
        import OpenGL.GL as gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dynamic = dotdict()

        # Render GS
        if self.render_network:
            # Render image here
            image = self.custom_render(self.camera.to_batch())
            if self.exposure != 1.0 or self.offset != 0.0:
                image = torch.cat([(image[..., :3] * self.exposure + self.offset), image[..., -1:]], dim=-1)  # add manual correction
            image = (image.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform

            self.quad.copy_to_texture(image)
            self.quad.draw()  # draw is typically faster by 0.5ms

        # Render meshes (or point clouds)
        if self.render_meshes:
            for mesh in self.meshes:
                mesh.render(self.camera)

        self.draw_imgui()  # defines GUI elements
        self.show_imgui()

    def draw_rendering_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):
        # Other rendering options like visualization type
        if imgui.collapsing_header('Rendering'):
            self.visualize_axes = imgui_toggle.toggle('Visualize axes', self.visualize_axes, config=self.static.toggle_ios_style)[1]
            self.visualize_bounds = imgui_toggle.toggle('Visualize bounds', self.visualize_bounds, config=self.static.toggle_ios_style)[1]
            self.visualize_cameras = imgui_toggle.toggle('Visualize cameras', self.visualize_cameras, config=self.static.toggle_ios_style)[1]

    def draw_model_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):
        pass

    def draw_imgui(self):
        from easyvolcap.utils.gl_utils import Mesh, Splat, Gaussian

        # Initialization
        glfw.poll_events()  # process pending events, keyboard and stuff
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()
        imgui.push_font(self.default_font)

        self.static.playing_time = self.camera_path.playing_time  # Remember this, if changed, update camera
        self.static.slider_width = imgui.get_window_width() * 0.65  # https://github.com/ocornut/imgui/issues/267
        self.static.toggle_ios_style = imgui_toggle.ios_style(size_scale=0.2)

        # Titles
        fps, frame_time = self.get_fps_and_frame_time()
        name, device, memory = self.get_device_and_memory()
        # glfw.set_window_title(self.window, self.window_title.format(FPS=fps)) # might confuse window managers
        self.static.fps = fps
        self.static.frame_time = frame_time
        self.static.name = name
        self.static.device = device
        self.static.memory = memory

        # Being the main window
        imgui.begin(f'{self.W}x{self.H} {fps:.3f} fps###main', flags=imgui.WindowFlags_.menu_bar)

        self.custom_gui()

        # End of main window and rendering
        imgui.end()

        imgui.pop_font()
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def custom_gui(self):
        self.draw_menu_gui()
        self.draw_banner_gui()
        self.draw_camera_gui()
        self.draw_rendering_gui()
        self.draw_keyframes_gui()
        self.draw_model_gui()
        self.draw_mesh_gui()
        self.draw_debug_gui()

    def custom_render(self, batch: dotdict):
        batch = to_cuda(batch)
        H, W = batch.meta.H.item(), batch.meta.W.item()
        return torch.rand(H, W, 4, device='cuda')
