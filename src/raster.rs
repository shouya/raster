use std::{cmp::max, collections::HashMap, convert::TryInto, mem};

use nalgebra::{
  Matrix4, Norm, Orthographic3, Point3, Unit, Vector2, Vector3, Vector4,
};

use crate::{
  lerp::{lerp, lerp_closed_iter, Lerp},
  shader::{DiffuseShader, PureColor, Shader, ShaderContext},
  util::sorted_tuple3,
};

pub type Color = Vector4<f32>;

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

  pub fn put_depth(&mut self, coords: (i32, i32), d: f32) -> Option<()> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    self.buffer[idx as usize] = d;
    Some(())
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
  inv_transform: Matrix4<f32>,
  perspective: Matrix4<f32>,
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
    // x,y,z^-1
    let relative_world = self.inv_transform.transform_point(&point);
    let mut point = self.perspective.transform_point(&relative_world);
    point.z = 1.0 / point.z;
    point
  }

  pub fn matrix(&self) -> Matrix4<f32> {
    self.perspective * self.inv_transform
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

  pub fn vertices(&self) -> [Point3<f32>; 3] {
    self.vertices.clone()
  }
}

pub struct Mesh {
  transform: Matrix4<f32>,
  vertices: Vec<Point3<f32>>,
  // every edge (a,b) must have a < b
  edges: HashMap<usize, Vec<usize>>,
  shader: Box<dyn Shader>,
}

impl Mesh {
  pub fn new() -> Self {
    // let shader = PureColor::new(COLOR::rgba(1.0, 0.0, 0.0, 0.0));
    let shader = DiffuseShader::new(
      COLOR::rgba(1.0, 0.0, 0.0, 0.0),
      COLOR::rgba(1.0, 1.0, 1.0, 1.0),
      Point3::new(-5.0, 10.0, 0.0),
    );

    Self {
      transform: Matrix4::identity(),
      vertices: Default::default(),
      edges: Default::default(),
      shader: Box::new(shader),
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

  pub fn iter_meshes(&self) -> impl Iterator<Item = &Mesh> + '_ {
    self.meshes.iter()
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
  pub point: Point3<f32>,
  pub normal: Unit<Vector3<f32>>,
  pub color: Color,
  pub uv: Vector2<f32>,
}

impl ScreenPt {
  pub fn new(point: Point3<f32>) -> Self {
    Self {
      point,
      color: COLOR::rgba(1.0, 0.0, 0.0, 1.0),
      uv: Vector2::new(0.0, 0.0),
      normal: Unit::new_normalize(point.coords),
    }
  }

  pub fn x(&self) -> f32 {
    self.point.x
  }
  pub fn y(&self) -> f32 {
    self.point.y
  }

  pub fn x_int(&self) -> i32 {
    self.x().round() as i32
  }

  pub fn y_int(&self) -> i32 {
    self.y().round() as i32
  }

  pub fn coords(&self) -> (i32, i32) {
    (self.x_int(), self.y_int())
  }

  pub fn depth(&self) -> f32 {
    self.point.z
  }
}

impl Lerp for ScreenPt {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    ScreenPt {
      point: lerp(t, &self.point, &other.point),
      normal: self.normal.slerp(&other.normal, t),
      color: self.color.lerp(&other.color, t),
      uv: self.uv.lerp(&other.uv, t),
    }
  }
}

pub struct Rasterizer {
  mode: RasterizerMode,
  view: Matrix4<f32>,
  image: Image,
  zbuffer: Zbuffer,
}

impl Rasterizer {
  pub fn new(size: (usize, usize)) -> Self {
    let image = Image::new(size);
    let zbuffer = Zbuffer::new(size);
    let mode = RasterizerMode::Shaded;
    let view =
      Orthographic3::new(0.0, size.0 as f32, size.1 as f32, 0.0, 0.0, -1.0)
        .inverse();

    Self {
      image,
      zbuffer,
      mode,
      view,
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
    for mesh in scene.iter_meshes() {
      for trig in mesh.iter_triangles() {
        self.draw_triangle_wireframe(
          camera,
          &trig,
          &mesh.transform,
          mesh.shader.as_ref(),
        );
      }
    }
  }
  pub fn rasterize_shaded(&mut self, scene: &Scene) {
    let camera = scene.camera();
    for mesh in scene.iter_meshes() {
      for trig in mesh.iter_triangles() {
        self.fill_triangle(
          camera,
          &trig,
          &mesh.transform,
          mesh.shader.as_ref(),
        );
      }
    }
  }

  pub fn into_image(self) -> Image {
    match self.mode {
      RasterizerMode::Zbuffer => self.zbuffer.to_image(),
      _ => self.image,
    }
  }

  // note: the output may go out of screen
  pub fn world_to_screen(
    &self,
    camera: &Camera,
    point: &Point3<f32>,
  ) -> ScreenPt {
    //
    // todo: fix the depth
    let pcam = camera.world_to_camera(&point);
    let pscr = self.view.transform_point(&pcam);
    ScreenPt::new(pscr)
  }

  pub fn size(&self) -> (usize, usize) {
    (self.image.width(), self.image.height())
  }

  // checks the zbuffer
  fn draw_pixel(
    &mut self,
    p: &ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    // beyond the camera clipping plane
    // if p.depth() < 0.0 || p.depth() > 1.0 {
    //   return;
    // }
    match self.zbuffer.depth(p.coords()) {
      None => return,
      Some(d) if p.depth() > *d => return,
      Some(_) => {
        let mut pt = p.clone();
        shader.fragment(context, &mut pt);
        self.put_pixel(pt.coords(), pt.color);
        self.zbuffer.put_depth(pt.coords(), p.depth());
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

  fn draw_triangle_wireframe(
    &mut self,
    camera: &Camera,
    triangle: &Triangle,
    model: &Matrix4<f32>,
    shader: &dyn Shader,
  ) {
    let context = ShaderContext {
      view: self.view.clone(),
      camera: camera.matrix(),
      model: model.clone(),
    };

    let mut pts: Vec<ScreenPt> = Vec::new();
    for point in triangle.vertices() {
      let mut pt = ScreenPt::new(point);
      shader.vertex(&context, &mut pt);
      pts.push(pt);
    }

    self.draw_line(&pts[1], &pts[2], &context, shader);
    self.draw_line(&pts[2], &pts[0], &context, shader);
    self.draw_line(&pts[0], &pts[1], &context, shader);
  }

  fn draw_line(
    &mut self,
    p1: &ScreenPt,
    p2: &ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    let dx = p2.x_int() - p1.x_int();
    let dy = p2.y_int() - p1.y_int();
    let n = max(dx.abs(), dy.abs());

    for pt in lerp_closed_iter(p1, p2, n as usize) {
      self.draw_pixel(&pt, context, shader);
    }
  }

  // fill a triangle that is flat at bottom
  fn fill_upper_triangle(
    &mut self,
    top: ScreenPt,
    bottom_left: ScreenPt,
    bottom_right: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    // ensure bottom is flat
    assert!(bottom_left.y_int() == bottom_right.y_int());
    // ensure top is above bottom
    assert!(top.y_int() <= bottom_left.y_int());
    // ensure bottom left is on the left
    // assert!(bottom_left.x <= bottom_right.x);

    let h = (bottom_left.y_int() - top.y_int()) as usize;

    let left_pts_iter = lerp_closed_iter(&top, &bottom_left, h);
    let right_pts_iter = lerp_closed_iter(&top, &bottom_right, h);

    for (l, r) in left_pts_iter.zip(right_pts_iter) {
      self.draw_horizontal_line(l, r, context, shader)
    }
  }

  fn fill_lower_triangle(
    &mut self,
    top_left: ScreenPt,
    top_right: ScreenPt,
    bottom: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    // ensure top is flat
    assert!(top_left.y_int() == top_right.y_int());
    // ensure top is above bottom
    assert!(top_left.y_int() <= bottom.y_int());
    // ensure top left is on the left
    // assert!(top_left.x <= right_right.x);

    let h = (bottom.y_int() - top_left.y_int()) as usize;
    // non-zero h
    let left_pts_iter = lerp_closed_iter(&top_left, &bottom, h);
    let right_pts_iter = lerp_closed_iter(&top_right, &bottom, h);

    for (l, r) in left_pts_iter.zip(right_pts_iter) {
      self.draw_horizontal_line(l, r, context, shader)
    }
  }

  fn fill_triangle(
    &mut self,
    camera: &Camera,
    triangle: &Triangle,
    model: &Matrix4<f32>,
    shader: &dyn Shader,
  ) {
    let context = ShaderContext {
      view: self.view.clone(),
      camera: camera.matrix(),
      model: model.clone(),
    };
    let mut pts: Vec<ScreenPt> = Vec::new();

    for point in triangle.vertices() {
      let mut pt = ScreenPt::new(point);
      shader.vertex(&context, &mut pt);
      pts.push(pt);
    }

    // if pts.iter().all(|p| self.invisible(p)) {
    //   // all out of range or occluded by closer objects
    //   return;
    // }

    let [upper, lower] =
      Self::horizontally_split_triangle(pts.as_mut_slice().try_into().unwrap());

    if let Some([a, b, c]) = upper {
      self.fill_upper_triangle(a, b, c, &context, shader);
    }

    if let Some([a, b, c]) = lower {
      self.fill_lower_triangle(a, b, c, &context, shader);
    }
  }

  //
  fn horizontally_split_triangle(
    pts: &mut [ScreenPt; 3],
  ) -> [Option<[ScreenPt; 3]>; 2] {
    pts.sort_unstable_by_key(|p| p.y_int());

    if pts[0].y_int() == pts[2].y_int() {
      // just a flat line
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    if pts[0].y_int() == pts[1].y_int() {
      // a lower triangle
      let lower_trig = [pts[0], pts[1], pts[2]];
      return [None, Some(lower_trig)];
    }

    if pts[1].y_int() == pts[2].y_int() {
      // a lower triangle
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    // a normal triangle that we need to split
    let t = (pts[1].y_int() - pts[0].y_int()) as f32
      / (pts[2].y_int() - pts[0].y_int()) as f32;
    let ptl = lerp(t, &pts[0], &pts[2]);
    let ptr = pts[1];

    let upper_trig = [pts[0], ptl, ptr];
    let lower_trig = [ptl, ptr, pts[2]];

    [Some(upper_trig), Some(lower_trig)]
  }

  fn draw_horizontal_line(
    &mut self,
    p1: ScreenPt,
    p2: ScreenPt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    let w = (p1.x_int() - p2.x_int()).abs() as usize;
    for p in lerp_closed_iter(&p1, &p2, w) {
      self.draw_pixel(&p, context, shader);
    }
  }

  fn out_of_view(&self, point: &ScreenPt) -> bool {
    let (x, y) = point.coords();
    let (w, h) = self.size();
    x < 0 || y < 0 || x >= w as i32 || y >= h as i32
  }

  fn invisible(&self, point: &ScreenPt) -> bool {
    if self.out_of_view(point) {
      return true;
    }

    let d = point.depth();
    if let Some(depth) = self.zbuffer.depth(point.coords()) {
      return d >= *depth;
    }

    unreachable!()
  }
}

#[cfg(test)]
mod test {
  use std::f32::consts::PI;

  use super::*;

  #[test]
  fn test_camera() {
    let aspect = 16.0 / 9.0;
    let fov = 130.0 / 180.0 * PI;
    let znear = 1.0;
    let zfar = 10.0;
    let perspective = Matrix4::new_perspective(aspect, fov, znear, zfar);
    for z in -10..10 {
      let pt = Point3::new(1.0, 2.0, z as f32);
      dbg!(z);
      dbg!(1.0 / perspective.transform_point(&pt).z);
    }

    assert!(false);
  }
}
