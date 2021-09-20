use eframe::{NativeOptions, egui::Vec2};

mod lerp;
mod raster;
mod shader;
mod util;
mod wavefront;
mod app;

use app::{render_scene, sample_scene};

fn main() {
  let args: Vec<_> = std::env::args().collect();
  if args.len() > 1 && args[1] == "bench" {
    bench_render();
  } else {
    let mut options: NativeOptions = Default::default();
    options.initial_window_size = Some(Vec2::new(900.0, 600.0));
    eframe::run_native(Box::new(app::RasterApp::default()), options);
  }
}

fn bench_render() {
  let tun = app::Tunable::default();
  let mut cache = app::SceneCache::new();
  const N: usize = 20;

  for _ in 0..N {
    let scene = sample_scene(&tun, &mut cache);
    render_scene(&tun, (600, 400), &scene);
  }
}
