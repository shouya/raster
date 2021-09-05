use std::{
  collections::HashMap,
  convert::TryInto,
  mem::{self, swap},
};

use nalgebra::{Matrix4, Point2, Point3, Vector2, Vector3, Vector4};

use crate::util::{point2_to_pixel, sorted_tuple3};

type Color = Vector4<f32>;

pub(crate) mod COLOR {
  use super::*;

  pub const fn rgb(r: f32, g: f32, b: f32) -> Color {
    rgba(r, g, b, 1.0)
  }
  pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
    Vector4::new(r, g, b, a)
  }
}

pub struct Image {
  dimension: (usize, usize),
  pixels: Vec<Color>,
}

impl Image {
  pub fn new(size: (usize, usize)) -> Self {
    let len = size.0 * size.1;
    let mut buffer = Vec::with_capacity(len);
    let clear_color = Vector4::new(0.0, 0.0, 0.0, 1.0);
    for _ in 0..len {
      buffer.push(clear_color);
    }

    Self {
      dimension: size,
      pixels: buffer,
    }
  }

  pub fn pixels(&self) -> impl Iterator<Item = &Color> {
    self.pixels.iter()
  }

  pub fn width(&self) -> usize {
    self.dimension.0
  }

  pub fn height(&self) -> usize {
    self.dimension.1
  }

  pub fn pixel(&self, coords: (i32, i32)) -> Option<&Color> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    self.pixels.get(idx as usize)
  }

  pub fn pixel_mut(&mut self, coords: (i32, i32)) -> Option<&mut Color> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    self.pixels.get_mut(idx as usize)
  }
}

pub struct Zbuffer {
  dimension: (usize, usize),
  // -1: unoccupied
  // 0..inf: distance
  buffer: Vec<f32>,
}

impl Zbuffer {
  pub fn new(size: (usize, usize)) -> Self {
    let len = size.0 * size.1;
    let mut buffer = Vec::with_capacity(len);
    for _ in 0..len {
      buffer.push(99999.0);
    }

    Self {
      dimension: size,
      buffer,
    }
  }

  pub fn width(&self) -> usize {
    self.dimension.0
  }

  pub fn height(&self) -> usize {
    self.dimension.1
  }

  pub fn depth(&self, coords: (i32, i32)) -> Option<&f32> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }

    let idx = coords.1 * self.width() as i32 + coords.0;
    self.buffer.get(idx as usize)
  }

  pub fn depth_mut(&mut self, coords: (i32, i32)) -> Option<&mut f32> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }

    let idx = coords.1 * self.width() as i32 + coords.0;
    self.buffer.get_mut(idx as usize)
  }

  pub fn to_image(&self) -> Image {
    let mut img = Image::new(self.dimension);
    for x in 0..self.width() {
      for y in 0..self.height() {
        let coords = (x as i32, y as i32);
        let d = *self.depth(coords).unwrap();
        *img.pixel_mut(coords).unwrap() = COLOR::rgb(d, d, d);
      }
    }
    img
  }
}

pub struct Camera {
  // world coordinate
  transform: Matrix4<f32>,
  perspective: Matrix4<f32>, // clipping_near: f32,
                             // clipping_far: f32
}

impl Camera {
  pub fn new_perspective(
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
  ) -> Self {
    let perspective = Matrix4::new_perspective(aspect, fovy, znear, zfar);
    let transform = Matrix4::identity();
    Self {
      perspective,
      transform,
    }
  }

  pub fn world_to_camera(&self, point: &Point3<f32>) -> Point3<f32> {
    // xy: position
    // z: depth
    (self.perspective * self.transform).transform_point(point)
  }

  pub fn transform(&mut self, trans: &Matrix4<f32>) {
    self.transform *= trans;
  }
}

pub struct Triangle {
  vertices: [Point3<f32>; 3],
}

impl From<(Point3<f32>, Point3<f32>, Point3<f32>)> for Triangle {
  fn from(tup: (Point3<f32>, Point3<f32>, Point3<f32>)) -> Self {
    Self {
      vertices: [tup.0, tup.1, tup.2],
    }
  }
}

impl Triangle {
  pub fn edges(&self) -> impl Iterator<Item = (Point3<f32>, Point3<f32>)> + '_ {
    vec![
      (self.vertices[0], self.vertices[1]),
      (self.vertices[1], self.vertices[2]),
      (self.vertices[2], self.vertices[0]),
    ]
    .into_iter()
  }
}

pub struct Mesh {
  transform: Matrix4<f32>,
  vertices: Vec<Point3<f32>>,
  // every edge (a,b) must have a < b
  edges: HashMap<usize, Vec<usize>>,
}

impl Mesh {
  pub fn new() -> Self {
    Self {
      transform: Matrix4::identity(),
      vertices: Default::default(),
      edges: Default::default(),
    }
  }

  pub fn new_trig(vertices: [Point3<f32>; 3]) -> Self {
    let mut mesh = Self::new();
    mesh.add_trig(&vertices);
    mesh
  }

  // a square with side length 2 on xy plane
  pub fn new_square() -> Self {
    let mut mesh = Self::new();
    let vertices: [Point3<f32>; 4] = [
      // top left
      [-1.0, 1.0, 0.0].into(),
      // bottom left
      [-1.0, -1.0, 0.0].into(),
      // bottom right
      [1.0, 1.0, 0.0].into(),
      // top right
      [1.0, -1.0, 0.0].into(),
    ];
    mesh.add_trig(&vertices[0..3].try_into().unwrap());
    mesh.add_trig(&vertices[1..4].try_into().unwrap());
    mesh
  }

  // a cube with side length 2
  pub fn new_cube() -> Self {
    let mut mesh = Self::new();
    let v: [Point3<f32>; 8] = [
      [1.0, 1.0, 1.0].into(),    // 0
      [1.0, 1.0, -1.0].into(),   // 1
      [1.0, -1.0, -1.0].into(),  // 2
      [1.0, -1.0, 1.0].into(),   // 3
      [-1.0, -1.0, 1.0].into(),  // 4
      [-1.0, -1.0, -1.0].into(), // 5
      [-1.0, 1.0, -1.0].into(),  // 6
      [-1.0, 1.0, 1.0].into(),   // 7
    ];

    // front
    mesh.add_quad(&[v[0], v[3], v[7], v[4]]);
    // back
    mesh.add_quad(&[v[1], v[2], v[6], v[5]]);
    // top
    mesh.add_quad(&[v[0], v[1], v[7], v[6]]);
    // bottom
    mesh.add_quad(&[v[2], v[3], v[5], v[4]]);
    // left
    mesh.add_quad(&[v[4], v[5], v[7], v[6]]);
    // right
    mesh.add_quad(&[v[0], v[1], v[3], v[2]]);
    mesh
  }

  pub fn transformed(mut self, transform: Matrix4<f32>) -> Self {
    self.transform = self.transform * transform;
    self
  }

  pub fn iter_triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
    let v = move |i| self.transform.transform_point(&self.vertices[i]);
    self
      .edges
      .keys()
      .flat_map(move |i| self.triangles_from_vert(*i))
      .map(move |(a, b, c)| (v(a), v(b), v(c)).into())
  }

  fn triangles_from_vert(
    &self,
    i: usize,
  ) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    self.edges[&i].iter().flat_map(move |a| {
      self.edges[&i].iter().flat_map(move |b| {
        if b <= a {
          vec![].into_iter()
        } else if self.has_edge(*a, *b) {
          let trig = sorted_tuple3((i, *a, *b)).into();
          vec![trig].into_iter()
        } else {
          vec![].into_iter()
        }
      })
    })
  }

  fn has_edge(&self, mut a: usize, mut b: usize) -> bool {
    if a == b {
      return false;
    }
    if a > b {
      mem::swap(&mut a, &mut b);
    }
    if !self.edges.contains_key(&a) {
      return false;
    }
    self.edges[&a].binary_search(&b).is_ok()
  }

  fn add_trig(&mut self, trig: &[Point3<f32>; 3]) {
    let a_idx = self.add_vert(trig[0]);
    let b_idx = self.add_vert(trig[1]);
    let c_idx = self.add_vert(trig[2]);
    self.add_edge(a_idx, b_idx);
    self.add_edge(b_idx, c_idx);
    self.add_edge(c_idx, a_idx);
  }

  fn add_quad(&mut self, quad: &[Point3<f32>; 4]) {
    self.add_trig(&[quad[0], quad[1], quad[2]]);
    self.add_trig(&[quad[1], quad[2], quad[3]]);
  }

  fn add_vert(&mut self, p: Point3<f32>) -> usize {
    if let Some(i) = self.vertices.iter().position(|x| *x == p) {
      return i;
    }

    let i = self.vertices.len();
    self.vertices.push(p);
    i
  }

  fn add_edge(&mut self, mut a: usize, mut b: usize) {
    assert!(a < self.vertices.len());
    assert!(b < self.vertices.len());
    if a > b {
      mem::swap(&mut a, &mut b);
    }
    let pts = self.edges.entry(a).or_default();
    match pts.binary_search(&b) {
      Ok(_i) => return,
      Err(i) => pts.insert(i, b),
    }
  }
}

pub struct Scene {
  camera: Camera,
  meshes: Vec<Mesh>,
}

impl Scene {
  pub fn new(camera: Camera) -> Self {
    Self {
      camera,
      meshes: vec![],
    }
  }

  pub fn add_mesh(&mut self, mesh: Mesh) {
    self.meshes.push(mesh);
  }

  pub fn iter_triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
    self.meshes.iter().flat_map(move |m| m.iter_triangles())
  }

  pub fn camera(&self) -> &Camera {
    &self.camera
  }
}

#[derive(PartialEq, Clone, Copy)]
pub enum RasterizerMode {
  Wireframe,
  Shaded,
  Zbuffer,
}

impl Default for RasterizerMode {
  fn default() -> Self {
    RasterizerMode::Shaded
  }
}

pub struct Rasterizer {
  mode: RasterizerMode,
  image: Image,
  zbuffer: Zbuffer,
}

impl Rasterizer {
  pub fn new(size: (usize, usize)) -> Self {
    let image = Image::new(size);
    let zbuffer = Zbuffer::new(size);
    let mode = RasterizerMode::Shaded;
    Self {
      image,
      zbuffer,
      mode,
    }
  }

  pub fn set_mode(&mut self, mode: RasterizerMode) {
    self.mode = mode;
  }

  pub fn rasterize(&mut self, scene: &Scene) {
    let camera = scene.camera();
    for trig in scene.iter_triangles() {
      self.draw_triangle(camera, &trig);
    }
  }

  pub fn into_image(self) -> Image {
    match self.mode {
      RasterizerMode::Zbuffer => self.zbuffer.to_image(),
      _ => self.image,
    }
  }

  pub fn draw_triangle(&mut self, camera: &Camera, triangle: &Triangle) {
    for (a, b) in triangle.edges() {
      let (a, da) = self.world_to_screen(camera, &a);
      let (b, db) = self.world_to_screen(camera, &b);
      self.draw_line(a, da, b, db);
    }
  }

  // point and depth
  pub fn world_to_screen(
    &self,
    camera: &Camera,
    point: &Point3<f32>,
  ) -> ((i32, i32), f32) {
    let pcam = camera.world_to_camera(&point);
    let pscr = self.camera_to_screen(pcam);
    (pscr, pcam.z)
  }

  // note: the output may go out of screen
  pub fn camera_to_screen(&self, point: Point3<f32>) -> (i32, i32) {
    let size = self.size();
    let scale = Vector2::new(size.0 as f32, size.1 as f32);
    let offset = Vector2::new(1.0, 1.0);
    let mapped_point = (point.xy().coords + offset / 2.0).component_mul(&scale);
    point2_to_pixel(&mapped_point.into())
  }

  pub fn size(&self) -> (usize, usize) {
    (self.image.width(), self.image.height())
  }

  // checks the zbuffer
  fn draw_pixel(&mut self, depth: f32, coords: (i32, i32), color: Color) {
    if let Some(d) = self.zbuffer.depth_mut(coords) {
      if *d > depth {
        *d = depth;
        self.put_pixel(coords, color)
      }
    }
  }

  // do not check for zbuffer
  fn put_pixel(&mut self, coords: (i32, i32), color: Color) {
    if let Some(pixel) = self.image.pixel_mut(coords) {
      *pixel = color;
    }
  }

  // fn draw_triangle(&mut self, triangle: Triangle) {
  //   triangle
  // }

  fn draw_line(
    &mut self,
    mut p1: (i32, i32),
    mut d1: f32,
    mut p2: (i32, i32),
    mut d2: f32,
  ) {
    let mut dx: i32 = p2.0 - p1.0;
    let mut dy: i32 = p2.1 - p1.1;
    let mut dz: f32 = d2 - d1;

    if dx.abs() >= dy.abs() {
      if dx < 0 {
        mem::swap(&mut p1, &mut p2);
        mem::swap(&mut d1, &mut d2);
        dx = -dx;
        dy = -dy;
        dz = -dz;
      }

      let x0 = p1.0;
      let y0 = p1.1;
      let z0 = d1;
      let d = dy as f32 / dx as f32;

      for n in 0..dx {
        let x = x0 + n;
        let y = y0 + (d * (n as f32)).round() as i32;
        let z = z0 + dz * (n as f32);
        self.draw_pixel(z, (x, y), COLOR::rgb(1.0, 0.0, 0.0));
      }
    } else {
      if dy < 0 {
        mem::swap(&mut p1, &mut p2);
        mem::swap(&mut d1, &mut d2);
        dx = -dx;
        dy = -dy;
        dz = -dz;
      }

      let x0 = p1.0;
      let y0 = p1.1;
      let z0 = d1;
      let d = dx as f32 / dy as f32;

      for n in 0..dy {
        let x = x0 + (d * (n as f32)).round() as i32;
        let y = y0 + n;
        let z = z0 + dz * (n as f32);
        self.draw_pixel(z, (x, y), COLOR::rgb(1.0, 0.0, 0.0));
      }
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_camera() {
    assert!(false);
  }
}
