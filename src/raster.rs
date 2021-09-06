use std::{
  collections::HashMap,
  convert::TryInto,
  mem::{self, swap},
};

use nalgebra::{Matrix4, Point2, Point3, Vector2, Vector3, Vector4};

use crate::util::{lerp, lerp_int, point2_to_pixel, sorted_tuple3};

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
      buffer.push(10.0);
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
        let d = *self.depth(coords).unwrap() / 2.0;
        *img.pixel_mut(coords).unwrap() = COLOR::rgb(d, d, d);
      }
    }
    img
  }
}

pub struct Camera {
  // world coordinate
  inv_transform: Matrix4<f32>,
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
    let inv_transform = Matrix4::identity();
    Self {
      perspective,
      inv_transform,
    }
  }

  pub fn world_to_camera(&self, point: &Point3<f32>) -> Point3<f32> {
    // xyz
    (self.perspective * self.inv_transform).transform_point(point)
  }

  pub fn transformd(&mut self, trans: &Matrix4<f32>) {
    self.inv_transform *= trans.pseudo_inverse(0.0001).unwrap();
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

  pub fn points(&self) -> [Point3<f32>; 3] {
    self.vertices.clone()
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
    self.transform = transform * self.transform;
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

/// A point on screen with integer xy coordinates and floating depth (z)
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct ScreenPt {
  pub x: i32,
  pub y: i32,
  pub z: f32,
}

impl ScreenPt {
  pub fn new(coords: (i32, i32), depth: f32) -> Self {
    Self {
      x: coords.0,
      y: coords.1,
      z: depth,
    }
  }
  pub fn iter_horizontal<'a>(
    p1: &'a Self,
    p2: &'a Self,
  ) -> impl Iterator<Item = Self> + 'a {
    ScreenPtIterator::new_horizontal(p1, p2)
  }

  pub fn coords(&self) -> (i32, i32) {
    (self.x, self.y)
  }

  pub fn depth(&self) -> f32 {
    self.z
  }
}

pub struct ScreenPtIterator<'a> {
  end: &'a ScreenPt,
  curr: ScreenPt,
  dx: i32,
  dy: i32,
  dz: f32,
}

impl<'a> Iterator for ScreenPtIterator<'a> {
  type Item = ScreenPt;

  fn next(&mut self) -> Option<Self::Item> {
    if self.curr.x == self.end.x && self.curr.y == self.end.y {
      return None;
    }

    let curr = self.curr.clone();
    self.curr.x += self.dx;
    self.curr.y += self.dy;
    self.curr.z += self.dz;
    Some(curr)
  }
}

impl<'a> ScreenPtIterator<'a> {
  pub fn new_horizontal(start: &'a ScreenPt, end: &'a ScreenPt) -> Self {
    assert!(start.y == end.y);
    let dy = 0;
    let dx = if start.x < end.x { 1 } else { -1 };
    let steps = (end.x - start.x).abs();
    let stepsn0 = if steps == 0 { 1.0 } else { steps as f32 };
    let dz = (end.z - start.z) / stepsn0;
    let curr = start.clone();
    Self {
      curr,
      end,
      dx,
      dy,
      dz,
    }
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
    match self.mode {
      RasterizerMode::Shaded => self.rasterize_shaded(scene),
      RasterizerMode::Wireframe => self.rasterize_wireframe(scene),
      RasterizerMode::Zbuffer => self.rasterize_shaded(scene),
    }
  }

  pub fn rasterize_wireframe(&mut self, scene: &Scene) {
    let camera = scene.camera();
    for trig in scene.iter_triangles() {
      self.draw_triangle_wireframe(camera, &trig);
    }
  }
  pub fn rasterize_shaded(&mut self, scene: &Scene) {
    let camera = scene.camera();
    for trig in scene.iter_triangles() {
      self.fill_triangle(camera, &trig);
    }
  }

  pub fn into_image(self) -> Image {
    match self.mode {
      RasterizerMode::Zbuffer => self.zbuffer.to_image(),
      _ => self.image,
    }
  }

  // point and depth
  pub fn world_to_screen(
    &self,
    camera: &Camera,
    point: &Point3<f32>,
  ) -> ScreenPt {
    let pcam = camera.world_to_camera(&point);
    let pscr = self.camera_to_screen(pcam);
    ScreenPt::new(pscr, pcam.z)
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
  fn draw_pixel(&mut self, p: &ScreenPt, color: Color) {
    // beyond the camera clipping plane
    if let Some(d) = self.zbuffer.depth_mut(p.coords()) {
      if p.depth() < *d {
        *d = p.depth();
        self.put_pixel(p.coords(), color)
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

  fn draw_triangle_wireframe(&mut self, camera: &Camera, triangle: &Triangle) {
    for (a, b) in triangle.edges() {
      let pa = self.world_to_screen(camera, &a);
      let pb = self.world_to_screen(camera, &b);
      self.draw_line(pa, pb);
    }
  }

  fn draw_line(&mut self, mut p1: ScreenPt, mut p2: ScreenPt) {
    let mut dx: i32 = p2.x - p1.x;
    let mut dy: i32 = p2.y - p1.y;

    if dx.abs() >= dy.abs() {
      if dx < 0 {
        mem::swap(&mut p1, &mut p2);
        dx = -dx;
        dy = -dy;
      }

      let x0 = p1.x;
      let y0 = p1.y;
      let z0 = p1.z;
      let dy = dy as f32 / dx as f32;
      let dz = (p2.z - p1.z) / dx as f32;

      for n in 0..=dx {
        let p = ScreenPt {
          x: x0 + n,
          y: y0 + (dy * n as f32).round() as i32,
          z: z0 + dz * n as f32,
        };
        self.draw_pixel(&p, COLOR::rgb(1.0, 0.0, 0.0));
      }
    } else {
      if dy < 0 {
        mem::swap(&mut p1, &mut p2);
        dx = -dx;
        dy = -dy;
      }

      let x0 = p1.x;
      let y0 = p1.y;
      let z0 = p1.z;
      let dx = dx as f32 / dy as f32;
      let dz = (p2.z - p1.z) / dy as f32;

      for n in 0..=dy {
        let p = ScreenPt {
          x: x0 + (dx * n as f32).round() as i32,
          y: y0 + n,
          z: z0 + dz * n as f32,
        };
        self.draw_pixel(&p, COLOR::rgb(1.0, 0.0, 0.0));
      }
    }
  }

  // fill a triangle that is flat at bottom
  fn fill_upper_triangle(
    &mut self,
    top: ScreenPt,
    bottom_left: ScreenPt,
    bottom_right: ScreenPt,
  ) {
    // ensure bottom is flat
    assert!(bottom_left.y == bottom_right.y);
    // ensure top is above bottom
    assert!(top.y <= bottom_left.y);
    // ensure bottom left is on the left
    // assert!(bottom_left.x <= bottom_right.x);

    let h = bottom_left.y - top.y;
    // non-zero h
    let hn0 = if h == 0 { 1.0 } else { h as f32 };
    let dxl = (bottom_left.x - top.x) as f32 / hn0;
    let dxr = (bottom_right.x - top.x) as f32 / hn0;
    let dzl = (bottom_left.z - top.z) / hn0;
    let dzr = (bottom_right.z - top.z) / hn0;
    let y0 = top.y;
    let x0 = top.x;
    let z0 = top.z;

    for n in 0..=h {
      let y = y0 + n;
      let xl = x0 + (dxl * n as f32).round() as i32;
      let xr = x0 + (dxr * n as f32).round() as i32;
      let zl = z0 + dzl * n as f32;
      let zr = z0 + dzr * n as f32;
      let pl = ScreenPt { y, x: xl, z: zl };
      let pr = ScreenPt { y, x: xr, z: zr };
      self.draw_horizontal_line(pl, pr);
    }
  }

  fn fill_lower_triangle(
    &mut self,
    top_left: ScreenPt,
    top_right: ScreenPt,
    bottom: ScreenPt,
  ) {
    // ensure top is flat
    assert!(top_left.y == top_right.y);
    // ensure top is above bottom
    assert!(top_left.y <= bottom.y);
    // ensure top left is on the left
    // assert!(top_left.x <= right_right.x);

    let h = bottom.y - top_left.y;
    // non-zero h
    let hn0 = if h == 0 { 1.0 } else { h as f32 };
    let dxl = (bottom.x - top_left.x) as f32 / hn0;
    let dxr = (bottom.x - top_right.x) as f32 / hn0;
    let dzl = (bottom.z - top_left.z) / hn0;
    let dzr = (bottom.z - top_right.z) / hn0;
    let y0 = bottom.y;
    let x0 = bottom.x;
    let z0 = bottom.z;

    for n in 0..=h {
      let y = y0 - n;
      let xl = x0 - (dxl * n as f32).round() as i32;
      let xr = x0 - (dxr * n as f32).round() as i32;
      let zl = z0 - dzl * n as f32;
      let zr = z0 - dzr * n as f32;
      let pl = ScreenPt { y, x: xl, z: zl };
      let pr = ScreenPt { y, x: xr, z: zr };
      self.draw_horizontal_line(pl, pr);
    }
  }

  fn fill_triangle(&mut self, camera: &Camera, triangle: &Triangle) {
    let mut pts: Vec<ScreenPt> = triangle
      .points()
      .iter()
      .map(|p| self.world_to_screen(camera, p))
      .collect();

    let [upper, lower] =
      Self::horizontally_split_triangle(pts.as_mut_slice().try_into().unwrap());

    if let Some([a, b, c]) = upper {
      self.fill_upper_triangle(a, b, c);
    }

    if let Some([a, b, c]) = lower {
      self.fill_lower_triangle(a, b, c);
    }
  }

  //
  fn horizontally_split_triangle(
    pts: &mut [ScreenPt; 3],
  ) -> [Option<[ScreenPt; 3]>; 2] {
    pts.sort_unstable_by_key(|p| p.y);

    if pts[0].y == pts[2].y {
      // just a flat line
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    if pts[0].y == pts[1].y {
      // a lower triangle
      let lower_trig = [pts[0], pts[1], pts[2]];
      return [None, Some(lower_trig)];
    }

    if pts[1].y == pts[2].y {
      // a lower triangle
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    // a normal triangle that we need to split
    let y = pts[1].y;
    let r = (pts[1].y - pts[0].y) as f32 / (pts[2].y - pts[0].y) as f32;
    let xl = lerp_int(r, pts[0].x, pts[2].x);
    let zl = lerp(r, pts[0].z, pts[2].z);
    let xr = pts[1].x;
    let zr = pts[1].z;

    let mut ptl = ScreenPt { y, x: xl, z: zl };
    let mut ptr = ScreenPt { y, x: xr, z: zr };
    if ptr.x < ptl.x {
      mem::swap(&mut ptr, &mut ptl);
    }
    let upper_trig = [pts[0], ptl, ptr];
    let lower_trig = [ptl, ptr, pts[2]];

    [Some(upper_trig), Some(lower_trig)]
  }

  fn draw_horizontal_line(&mut self, p1: ScreenPt, p2: ScreenPt) {
    for pt in ScreenPt::iter_horizontal(&p1, &p2) {
      self.draw_pixel(&pt, COLOR::rgb(1.0, 0.0, 0.0))
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
