use std::{cmp::max, convert::TryInto};

use approx::abs_diff_eq;
use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};

use crate::{
  lerp::{lerp, lerp_closed_iter, Lerp},
  shader::{Light, Shader, ShaderContext, SimpleMaterial},
  util::f32_cmp,
};

pub type Color = Vector4<f32>;

#[allow(non_snake_case)]
pub(crate) mod COLOR {
  use super::*;

  pub const fn black() -> Color {
    rgba(0.0, 0.0, 0.0, 1.0)
  }

  pub const fn rgb(r: f32, g: f32, b: f32) -> Color {
    rgba(r, g, b, 1.0)
  }
  pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
    Vector4::new(r, g, b, a)
  }
}

#[derive(Debug, Clone)]
pub struct Image<T> {
  dimension: (usize, usize),
  pixels: Vec<T>,
}

impl<T> Image<T> {
  pub fn new(size: (usize, usize)) -> Self
  where
    T: Default,
  {
    let len = size.0 * size.1;
    let mut buffer = Vec::with_capacity(len);
    for _ in 0..len {
      buffer.push(Default::default());
    }

    Self {
      dimension: size,
      pixels: buffer,
    }
  }

  pub fn new_filled(size: (usize, usize), val: &T) -> Self
  where
    T: Clone,
  {
    let len = size.0 * size.1;
    let mut buffer = Vec::with_capacity(len);
    for _ in 0..len {
      buffer.push(val.clone());
    }

    Self {
      dimension: size,
      pixels: buffer,
    }
  }

  pub fn pixels(&self) -> impl Iterator<Item = &T> {
    self.pixels.iter()
  }

  pub fn width(&self) -> usize {
    self.dimension.0
  }

  pub fn height(&self) -> usize {
    self.dimension.1
  }

  pub fn pixel(&self, coords: (i32, i32)) -> Option<&T> {
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

  pub fn pixel_mut(&mut self, coords: (i32, i32)) -> Option<&mut T> {
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

  pub fn put_pixel(&mut self, coords: (i32, i32), value: T) {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    *self.pixels.get_mut(idx as usize).unwrap() = value;
  }

  pub fn map<F, S>(self, f: F) -> Image<S>
  where
    F: Fn(T) -> S,
  {
    let dimension = self.dimension;
    let pixels = self.pixels.into_iter().map(f).collect();
    Image { pixels, dimension }
  }
}

#[derive(Debug, Clone)]
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

  pub fn matrix(&self) -> Matrix4<f32> {
    self.perspective * self.inv_transform
  }

  pub fn transformd(&mut self, trans: &Matrix4<f32>) {
    self.inv_transform *= trans.pseudo_inverse(0.0001).unwrap();
  }
}

#[derive(Debug, Clone)]
pub struct Trig<T> {
  vertices: [T; 3],
}

impl<T> From<[T; 3]> for Trig<T> {
  fn from(vertices: [T; 3]) -> Self {
    Self { vertices }
  }
}

impl<'a, T> Trig<T> {
  pub fn a(&self) -> &T {
    &self.vertices[0]
  }
  pub fn b(&self) -> &T {
    &self.vertices[1]
  }
  pub fn c(&self) -> &T {
    &self.vertices[2]
  }

  #[allow(unused)]
  pub fn edges<S>(&'a self) -> impl Iterator<Item = (S, S)>
  where
    S: From<&'a T>,
  {
    vec![
      (From::from(&self.vertices[0]), From::from(&self.vertices[1])),
      (From::from(&self.vertices[1]), From::from(&self.vertices[2])),
      (From::from(&self.vertices[2]), From::from(&self.vertices[0])),
    ]
    .into_iter()
  }

  pub fn vertices<S>(&'a self) -> [S; 3]
  where
    T: Clone,
    S: From<T>,
  {
    [
      From::from(self.vertices[0].clone()),
      From::from(self.vertices[1].clone()),
      From::from(self.vertices[2].clone()),
    ]
  }

  pub fn as_ref(&self) -> Trig<&T> {
    Trig {
      vertices: [self.a(), self.b(), self.c()],
    }
  }

  pub fn map<S, F>(self, f: F) -> Trig<S>
  where
    F: Fn(T) -> S,
  {
    Trig {
      vertices: self.vertices.map(|x| f(x)),
    }
  }
}

#[derive(Debug, Clone)]
pub struct Face<T> {
  vertices: Vec<T>,
  double_faced: bool,
}

impl<T> Face<T> {
  pub fn new(double_faced: bool) -> Self {
    Face {
      vertices: Vec::new(),
      double_faced,
    }
  }
  pub fn add_vert(&mut self, v: T) {
    self.vertices.push(v);
  }

  pub fn vertices(&self) -> &[T] {
    self.vertices.as_ref()
  }

  pub fn double_faced(&self) -> bool {
    self.double_faced
  }

  pub fn as_ref(&self) -> Face<&T> {
    Face {
      vertices: self.vertices.iter().collect(),
      double_faced: self.double_faced,
    }
  }

  pub fn len(&self) -> usize {
    self.vertices.len()
  }

  pub fn triangulate(&self) -> impl Iterator<Item = Trig<T>>
  where
    T: Copy,
  {
    debug_assert!(self.vertices.len() >= 3);

    let mut res = Vec::new();

    let v0 = self.vertices[0];
    let vs = self.vertices.iter().skip(1).collect::<Vec<_>>();
    for v in vs.windows(2) {
      res.push(Trig::from([v0, *v[0], *v[1]]));
    }

    res.into_iter()
  }

  pub fn convert<S>(self) -> Face<S>
  where
    S: From<T>,
  {
    Face {
      vertices: self.vertices.into_iter().map(|x| x.into()).collect(),
      double_faced: self.double_faced,
    }
  }

  pub fn edges(&self) -> impl Iterator<Item = (T, T)> + '_
  where
    T: Copy,
  {
    let n = self.vertices.len();
    (0..n).map(move |i| {
      let a = self.vertices[i];
      let b = self.vertices[(i + 1) % n];
      (a, b)
    })
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.vertices.iter_mut().for_each(f)
  }
}

#[derive(Debug, Clone)]
pub struct IndexedPolyVert {
  vertex_index: usize,
  texture_index: Option<usize>,
  normal_index: Option<usize>,
}

impl IndexedPolyVert {
  pub fn new(vertex_index: usize) -> Self {
    Self {
      vertex_index,
      texture_index: None,
      normal_index: None,
    }
  }

  pub fn new_texture(vertex_index: usize, texture_index: usize) -> Self {
    Self {
      vertex_index,
      texture_index: Some(texture_index),
      normal_index: None,
    }
  }

  pub fn new_normal(vertex_index: usize, normal_index: usize) -> Self {
    Self {
      vertex_index,
      normal_index: Some(normal_index),
      texture_index: None,
    }
  }

  pub fn new_texture_normal(
    vertex_index: usize,
    texture_index: usize,
    normal_index: usize,
  ) -> Self {
    Self {
      vertex_index,
      texture_index: Some(texture_index),
      normal_index: Some(normal_index),
    }
  }
}

#[derive(Debug, Clone)]
pub struct PolyVert<'a> {
  pub vertex: &'a Point3<f32>,
  pub texture_coords: Option<&'a Vector2<f32>>,
  pub normal: Option<&'a Vector3<f32>>,
}

#[derive(Debug, Clone)]
pub struct Mesh {
  pub material: Option<SimpleMaterial>,
  pub transform: Matrix4<f32>,
  pub vertices: Vec<Point3<f32>>,
  pub vertex_normals: Vec<Vector3<f32>>,
  pub texture_coords: Vec<Vector2<f32>>,
  pub faces: Vec<Face<IndexedPolyVert>>,
  pub double_faced: bool,
}

impl Mesh {
  #[allow(unused)]
  pub fn new() -> Self {
    Self {
      transform: Matrix4::identity(),
      vertices: Default::default(),
      vertex_normals: Default::default(),
      texture_coords: Default::default(),
      faces: Vec::new(),
      material: None,
      double_faced: false,
    }
  }

  pub fn num_faces(&self) -> usize {
    self.faces.len()
  }

  pub fn double_faced(mut self, double_faced: bool) -> Self {
    self.double_faced = double_faced;
    self
  }

  pub fn faces(&self) -> impl Iterator<Item = Face<PolyVert<'_>>> {
    self.faces.iter().map(move |f| self.get_face(f))
  }

  pub fn get_face(&self, face: &Face<IndexedPolyVert>) -> Face<PolyVert<'_>> {
    let mut res = Face::new(self.double_faced);
    for vert in face.vertices() {
      let vertex = &self.vertices[vert.vertex_index];
      let texture_coords = vert.texture_index.map(|i| &self.texture_coords[i]);
      let normal = vert.normal_index.map(|i| &self.vertex_normals[i]);

      res.add_vert(PolyVert {
        vertex,
        texture_coords,
        normal,
      })
    }
    res
  }

  pub fn transformed(mut self, transform: Matrix4<f32>) -> Self {
    self.transform = transform * self.transform;
    self
  }

  pub fn shader(&self) -> &dyn Shader {
    self.material.as_ref().unwrap_or_else(|| SimpleMaterial::plaster())
  }
}

pub struct Scene {
  camera: Camera,
  lights: Vec<Light>,
  meshes: Vec<Mesh>,
}

impl Scene {
  pub fn new(camera: Camera) -> Self {
    Self {
      camera,
      lights: vec![],
      meshes: vec![],
    }
  }

  pub fn add_light(&mut self, light: Light) {
    self.lights.push(light);
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

  pub fn lights(&self) -> &[Light] {
    self.lights.as_slice()
  }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RasterizerMode {
  Wireframe,
  Shaded,
  Clipped,
}

impl Default for RasterizerMode {
  fn default() -> Self {
    RasterizerMode::Shaded
  }
}

/// A point on screen with integer xy coordinates and floating depth (z)
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Pt {
  pub clip_pos: Point3<f32>,
  pub world_pos: Point3<f32>,
  pub color: Color,
  pub normal: Vector3<f32>,
  pub uv: Vector2<f32>,
  pub buf_v2: Option<Vector2<f32>>,
  pub buf_v3: Option<Vector3<f32>>,
}

impl<'a> From<&PolyVert<'a>> for Pt {
  fn from(v: &PolyVert<'a>) -> Self {
    let mut pt = Self::new(*v.vertex);
    if let Some(uv) = v.texture_coords {
      pt.set_uv(*uv);
    }
    if let Some(normal) = v.normal {
      pt.set_normal(*normal);
    }
    pt
  }
}

impl Pt {
  pub fn new(point: Point3<f32>) -> Self {
    Self {
      clip_pos: point,
      world_pos: point,
      color: COLOR::rgba(1.0, 0.0, 0.0, 1.0),
      uv: Vector2::new(0.0, 0.0),
      normal: point.coords,
      buf_v2: None,
      buf_v3: None,
    }
  }

  pub fn set_uv(&mut self, uv: Vector2<f32>) {
    self.uv = uv;
  }

  pub fn set_normal(&mut self, normal: Vector3<f32>) {
    self.normal = normal.normalize();
  }

  pub fn clip_x(&self) -> f32 {
    self.clip_pos.x
  }
  pub fn clip_y(&self) -> f32 {
    self.clip_pos.y
  }

  pub fn depth(&self) -> f32 {
    self.clip_pos.z
  }
}

impl Lerp for Pt {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    if self == other {
      return *self;
    }

    Pt {
      clip_pos: lerp(t, &self.clip_pos, &other.clip_pos),
      world_pos: lerp(t, &self.world_pos, &other.world_pos),
      normal: lerp(t, &self.normal, &other.normal),
      color: lerp(t, &self.color, &other.color),
      uv: lerp(t, &self.uv, &other.uv),
      buf_v2: lerp(t, &self.buf_v2, &other.buf_v2),
      buf_v3: lerp(t, &self.buf_v3, &other.buf_v3),
    }
  }
}

#[derive(Debug, Default, Clone)]
pub struct RasterizerMetric {
  pub faces_rendered: usize,
  pub triangles_rendered: usize,
  pub clipped_triangles_rendered: usize,
  pub sub_triangles_rendered: usize,
  pub hidden_face_removed: usize,
  pub lines_drawn: usize,
  pub horizontal_lines_drawn: usize,
  pub vertices_shaded: usize,
  pub pixels_shaded: usize,
  pub pixels_discarded: usize,
}

pub struct Rasterizer {
  size: (f32, f32),
  mode: RasterizerMode,
  image: Image<Color>,
  zbuffer: Image<f32>,
  metric: RasterizerMetric,
}

impl Rasterizer {
  pub fn new(size: (usize, usize)) -> Self {
    let image = Image::new(size);
    let zbuffer = Image::new_filled(size, &1.0);
    let mode = RasterizerMode::Shaded;
    let size = (image.width() as f32, image.height() as f32);
    let metric = Default::default();

    Self {
      size,
      image,
      zbuffer,
      mode,
      metric,
    }
  }

  pub fn set_mode(&mut self, mode: RasterizerMode) {
    self.mode = mode;
  }

  pub fn rasterize(&mut self, scene: &Scene) {
    for mesh in scene.iter_meshes() {
      for face in mesh.faces() {
        let mut face: Face<Pt> = face.as_ref().convert();
        let context = self.shader_context(scene, mesh);
        let shader = mesh.shader();

        face.map_in_place(|mut p| shader.vertex(&context, &mut p));
        self.metric.vertices_shaded += face.len();

        if self.is_hidden_surface(&face) {
          self.metric.hidden_face_removed += 1;
          continue;
        }

        self.metric.faces_rendered += 1;
        self.rasterize_face(&mut face, &context, shader);
      }
    }
  }

  pub fn rasterize_face(
    &mut self,
    face: &mut Face<Pt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    use RasterizerMode::*;

    match self.mode {
      Shaded => {
        for trig in face.triangulate() {
          self.fill_triangle(&trig, &context, shader);
        }
      }
      Clipped => {
        for trig in face.triangulate() {
          self.draw_triangle_clipped(&trig, &context, shader);
        }
      }
      Wireframe => {
        for (a, b) in face.edges() {
          self.draw_line(a, b, &context, shader);
        }
      }
    }
  }

  fn clip_triangle(&self, _trig: &Trig<Pt>) -> Vec<Trig<Pt>> {
    vec![&_trig]
      .into_iter()
      .flat_map(|t| {
        self.clip_triangle_component(&t, |p| p.clip_pos.z, -1.0, -1.0)
      })
      .flat_map(|t| {
        self.clip_triangle_component(&t, |p| p.clip_pos.z, 1.0, 1.0)
      })
      .flat_map(|t| {
        self.clip_triangle_component(&t, |p| p.clip_pos.x, -1.0, -1.0)
      })
      .flat_map(|t| {
        self.clip_triangle_component(&t, |p| p.clip_pos.x, 1.0, 1.0)
      })
      .flat_map(|t| {
        self.clip_triangle_component(&t, |p| p.clip_pos.y, -1.0, -1.0)
      })
      .flat_map(|t| {
        self.clip_triangle_component(&t, |p| p.clip_pos.y, 1.0, 1.0)
      })
      .collect()
  }

  fn clip_triangle_component<F>(
    &self,
    trig: &Trig<Pt>,
    get_comp: F,
    lim: f32,
    sign: f32,
  ) -> Vec<Trig<Pt>>
  where
    F: Fn(&Pt) -> f32,
  {
    let (va, vb, vc) = (trig.a(), trig.b(), trig.c());
    let v: [f32; 3] = trig.as_ref().map(get_comp).vertices();
    let (a, b, c) = (v[0], v[1], v[2]);

    // redefine lt, gt, le, ge operator based on the sign
    let lt = |x: f32, y: f32| x * sign < y * sign;
    let ge = |x: f32, y: f32| x * sign >= y * sign;
    let in_lim = |x: f32| lt(x, lim);
    let out_lim = |x: f32| ge(x, lim);

    // case 1: all vertex within range
    if in_lim(a) && in_lim(b) && in_lim(c) {
      return vec![trig.clone()];
    }

    // case 2: all vertex out of range
    if out_lim(a) && out_lim(b) && out_lim(c) {
      // within the range; draw without clipping
      return vec![];
    }

    // case 3: two vertices out of range
    if in_lim(a) && out_lim(b) && out_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return vec![[*va, new_vb, new_vc].into()];
    }
    if out_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return vec![[new_va, *vb, new_vc].into()];
    }
    if out_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return vec![[new_va, new_vb, *vc].into()];
    }

    // case 4: one vertex out of range
    if out_lim(a) && in_lim(b) && in_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return vec![[*vb, *vc, new_vb].into(), [*vc, new_vc, new_vb].into()];
    }
    if in_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return vec![[*va, *vc, new_va].into(), [*vc, new_va, new_vc].into()];
    }
    if in_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return vec![[*va, *vb, new_va].into(), [*vb, new_va, new_vb].into()];
    }

    unreachable!()
  }

  // return false if the line is off-view and should be skipped
  fn clip_line(&self, a: &mut Pt, b: &mut Pt) -> bool {
    let get_x = |p: &Pt| p.clip_pos.x;
    let get_y = |p: &Pt| p.clip_pos.y;
    let get_z = |p: &Pt| p.clip_pos.z;

    if !Self::clip_line_component(a, b, get_x, -1.0, 1.0) {
      return false;
    }
    if !Self::clip_line_component(a, b, get_y, -1.0, 1.0) {
      return false;
    }
    if !Self::clip_line_component(a, b, get_z, -1.0, 1.0) {
      return false;
    }

    true
  }

  // return false if the line is off-view and should be skipped
  fn clip_line_component<F>(
    a: &mut Pt,
    b: &mut Pt,
    get_comp: F,
    min: f32,
    max: f32,
  ) -> bool
  where
    F: Fn(&Pt) -> f32,
  {
    let mut av = get_comp(a);
    let mut bv = get_comp(b);

    if av < min && bv < min {
      // both beyond min; skip this line
      return false;
    }
    if av > max && bv > max {
      // both beyond max; skip this line
      return false;
    }
    if av >= min && av <= max && bv >= min && bv <= max {
      // within the range; draw without clipping
      return true;
    }

    if av < min && bv >= min {
      // clip a on min
      let t = (min - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *a = lerp(t, a, b);
    } else if av >= min && bv < min {
      // clip b on min
      let t = (min - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *b = lerp(t, a, b);
    }

    // recalculate because a and b may be changed
    av = get_comp(a);
    bv = get_comp(b);

    if av > max && bv <= max {
      // clip a on max
      let t = (max - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *a = lerp(t, a, b);
    } else if av <= max && bv > max {
      // clip b on max
      let t = (max - av) / (bv - av);
      assert!((0.0..=1.0).contains(&t));
      *b = lerp(t, a, b);
    }

    true
  }

  pub fn to_coords(&self, pt: &Pt) -> (i32, i32) {
    let (w, h) = self.size_f32();
    let point = pt.clip_pos;
    let x = 0.5 * w * (point.x + 1.0);
    let y = 0.5 * h * (1.0 - point.y);
    (x as i32, y as i32)
  }

  pub fn to_x_coord(&self, x: f32) -> i32 {
    let (w, _h) = self.size_f32();
    (0.5 * (x + 1.0) * w) as i32
  }
  pub fn to_y_coord(&self, y: f32) -> i32 {
    let (_w, h) = self.size_f32();
    (0.5 * h * (1.0 - y)) as i32
  }

  pub fn zbuffer_image(&self) -> Image<Color> {
    let to_comp = |d| (d + 1.0) / 2.0;
    let to_color = |d| COLOR::rgb(to_comp(d), to_comp(d), to_comp(d));
    self.zbuffer.clone().map(to_color)
  }

  pub fn metric(&self) -> RasterizerMetric {
    self.metric.clone()
  }

  pub fn into_image(self) -> Image<Color> {
    self.image
  }

  pub fn size_f32(&self) -> (f32, f32) {
    self.size
  }

  // checks the zbuffer
  fn draw_pixel(
    &mut self,
    mut p: Pt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    let coords = self.to_coords(&p);

    match self.zbuffer.pixel(coords) {
      None => return,
      Some(d) if p.depth() > *d => {
        self.metric.pixels_discarded += 1;
        return;
      }
      Some(_) => {
        self.metric.pixels_shaded += 1;
        shader.fragment(context, &mut p);
        self.put_pixel(coords, p.color);
        self.zbuffer.put_pixel(coords, p.depth());
      }
    }
  }

  fn is_hidden_surface(&self, face: &Face<Pt>) -> bool {
    if face.double_faced() {
      return false;
    }

    let positive_direction: Vector3<f32> = [0.0, 0.0, 1.0].into();
    let v1 = face.vertices()[1].clip_pos - face.vertices()[0].clip_pos;
    let v2 = face.vertices()[2].clip_pos - face.vertices()[0].clip_pos;
    let normal = v1.cross(&v2);
    // no need to get the real normalized normal because we don't need
    // an exact number. The sum of the normal value is enough.
    normal.dot(&positive_direction) < 0.0
  }

  // do not check for zbuffer
  fn put_pixel(&mut self, coords: (i32, i32), color: Color) {
    if let Some(pixel) = self.image.pixel_mut(coords) {
      *pixel = color;
    }
  }

  fn draw_line(
    &mut self,
    mut p1: Pt,
    mut p2: Pt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    if !self.clip_line(&mut p1, &mut p2) {
      // out of screen
      return;
    }

    self.metric.lines_drawn += 1;

    let (p1x, p1y) = self.to_coords(&p1);
    let (p2x, p2y) = self.to_coords(&p2);
    let dx = p2x - p1x;
    let dy = p2y - p1y;
    let n = max(dx.abs(), dy.abs());
    for pt in lerp_closed_iter(&p1, &p2, n as usize) {
      self.draw_pixel(pt, context, shader);
    }
  }

  // fill a triangle that is flat at bottom
  fn fill_upper_triangle(
    &mut self,
    top: Pt,
    bottom_left: Pt,
    bottom_right: Pt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    self.metric.sub_triangles_rendered += 1;

    let top_y = self.to_y_coord(top.clip_y());
    let bottom_y = self.to_y_coord(bottom_left.clip_y());
    let h = (top_y - bottom_y).abs() as usize;

    let left_pts_iter = lerp_closed_iter(&top, &bottom_left, h);
    let right_pts_iter = lerp_closed_iter(&top, &bottom_right, h);

    for (l, r) in left_pts_iter.zip(right_pts_iter) {
      self.draw_horizontal_line(l, r, context, shader)
    }
  }

  fn fill_lower_triangle(
    &mut self,
    top_left: Pt,
    top_right: Pt,
    bottom: Pt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    self.metric.sub_triangles_rendered += 1;

    let top_y = self.to_y_coord(top_left.clip_y());
    let bottom_y = self.to_y_coord(bottom.clip_y());
    let h = (top_y - bottom_y).abs() as usize;
    // non-zero h
    let left_pts_iter = lerp_closed_iter(&top_left, &bottom, h);
    let right_pts_iter = lerp_closed_iter(&top_right, &bottom, h);

    for (l, r) in left_pts_iter.zip(right_pts_iter) {
      self.draw_horizontal_line(l, r, context, shader)
    }
  }

  fn shader_context<'a>(
    &self,
    scene: &'a Scene,
    mesh: &Mesh,
  ) -> ShaderContext<'a> {
    ShaderContext {
      camera: scene.camera().matrix(),
      model: mesh.transform.clone(),
      lights: scene.lights(),
    }
  }

  fn draw_triangle_clipped(
    &mut self,
    trig: &Trig<Pt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    for trig in self.clip_triangle(&trig) {
      let pts: Vec<_> = trig.vertices().into();
      self.draw_line(pts[0], pts[1], &context, shader);
      self.draw_line(pts[1], pts[2], &context, shader);
      self.draw_line(pts[2], pts[0], &context, shader);
    }
  }

  fn fill_triangle(
    &mut self,
    trig: &Trig<Pt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    self.metric.triangles_rendered += 1;
    for trig in self.clip_triangle(&trig) {
      self.metric.clipped_triangles_rendered += 1;
      let mut pts: Vec<_> = trig.vertices().into();

      let [upper, lower] = Self::horizontally_split_triangle(
        pts.as_mut_slice().try_into().unwrap(),
      );

      if let Some([a, b, c]) = upper {
        self.fill_upper_triangle(a, b, c, &context, shader);
      }

      if let Some([a, b, c]) = lower {
        self.fill_lower_triangle(a, b, c, &context, shader);
      }
    }
  }

  fn horizontally_split_triangle(pts: &mut [Pt; 3]) -> [Option<[Pt; 3]>; 2] {
    const EPS: f32 = 0.0001;
    pts.sort_unstable_by(|p1, p2| f32_cmp(&p1.clip_y(), &p2.clip_y()));

    if abs_diff_eq!(pts[0].clip_y(), pts[2].clip_y(), epsilon = EPS) {
      // just a flat line
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    if abs_diff_eq!(pts[0].clip_y(), pts[1].clip_y(), epsilon = EPS) {
      // a lower triangle
      let lower_trig = [pts[0], pts[1], pts[2]];
      return [None, Some(lower_trig)];
    }

    if abs_diff_eq!(pts[1].clip_y(), pts[2].clip_y(), epsilon = EPS) {
      // a lower triangle
      let upper_trig = [pts[0], pts[1], pts[2]];
      return [Some(upper_trig), None];
    }

    // a normal triangle that we need to split
    let t =
      (pts[1].clip_y() - pts[0].clip_y()) / (pts[2].clip_y() - pts[0].clip_y());
    let ptl = lerp(t, &pts[0], &pts[2]);
    let ptr = pts[1];

    let upper_trig = [pts[0], ptl, ptr];
    let lower_trig = [ptl, ptr, pts[2]];

    [Some(upper_trig), Some(lower_trig)]
  }

  fn draw_horizontal_line(
    &mut self,
    p1: Pt,
    p2: Pt,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    self.metric.horizontal_lines_drawn += 1;
    let x1 = self.to_x_coord(p1.clip_x());
    let x2 = self.to_x_coord(p2.clip_x());
    let w = (x1 - x2).abs() as usize;
    for p in lerp_closed_iter(&p1, &p2, w) {
      self.draw_pixel(p, context, shader);
    }
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
