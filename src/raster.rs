use std::{borrow::Cow, cmp::max, convert::TryInto, time::Instant};

use approx::abs_diff_eq;
use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};
use smallvec::{smallvec, SmallVec};

use crate::{
  lerp::{lerp, lerp_closed_iter, Lerp},
  shader::{
    Light, Shader, ShaderContext, ShaderOptions, SimpleMaterial, TextureStash,
  },
  util::f32_cmp,
  wavefront::Mesh,
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

  pub fn size(&self) -> (usize, usize) {
    self.dimension
  }
}

impl Image<Color> {
  pub fn nearest_color(&self, uv: Vector2<f32>) -> Color {
    let coords = self.nearest_coords(uv, (0, 0));
    *self.pixel(coords).unwrap()
  }

  pub fn bilinear_color(&self, uv: Vector2<f32>) -> Color {
    // row 1: |a b|
    // row 2: |c d|
    let color_a = *self.pixel(self.nearest_coords(uv, (0, 0))).unwrap();
    let color_b = *self.pixel(self.nearest_coords(uv, (1, 0))).unwrap();
    let color_c = *self.pixel(self.nearest_coords(uv, (0, 1))).unwrap();
    let color_d = *self.pixel(self.nearest_coords(uv, (1, 1))).unwrap();

    let x = uv.x * (self.width() - 1) as f32;
    let y = uv.y * (self.height() - 1) as f32;
    let xt = x - x.floor();
    let yt = y - y.floor();

    let color_l = lerp(yt, &color_a, &color_c);
    let color_r = lerp(yt, &color_b, &color_d);
    let color = lerp(xt, &color_l, &color_r);

    color
  }

  /// if out of bound, return the nearest coordinates within the bound
  fn nearest_coords(&self, uv: Vector2<f32>, offset: (i32, i32)) -> (i32, i32) {
    let mut x = (uv.x * (self.width() - 1) as f32) as i32 + offset.0;
    let mut y = (uv.y * (self.height() - 1) as f32) as i32 + offset.1;

    // if x < 0 {
    //   x = 0;
    // }
    // if y < 0 {
    //   y = 0;
    // }
    if x >= self.width() as i32 {
      x = self.width() as i32 - 1;
    }
    if y >= self.height() as i32 {
      y = self.height() as i32 - 1;
    }

    (x, y)
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

  pub fn project_point(&self, pt: &Point3<f32>) -> Point3<f32> {
    self.matrix().transform_point(pt)
  }
}

#[derive(Debug, Clone)]
pub struct Line<T> {
  ends: [T; 2],
}

impl<T> From<[T; 2]> for Line<T> {
  fn from(ends: [T; 2]) -> Self {
    Self { ends }
  }
}

impl<T> From<(T, T)> for Line<T> {
  fn from(ends: (T, T)) -> Self {
    Self {
      ends: [ends.0, ends.1],
    }
  }
}

impl<T> Line<T> {
  pub fn a(&self) -> &T {
    &self.ends[0]
  }
  pub fn b(&self) -> &T {
    &self.ends[1]
  }

  // The caller needs to ensure "self" is
  pub fn to_horizontal_pixels(self) -> impl Iterator<Item = T>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    let x1 = self.a().to_clip().x;
    let x2 = self.b().to_clip().x;
    let w = (x1 - x2).abs() as usize;

    lerp_closed_iter(self.ends[0], self.ends[1], w + 1)
  }

  pub fn to_pixels(self) -> impl Iterator<Item = T>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    let a = self.a().to_clip();
    let b = self.b().to_clip();
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let n = f32::max(dx.abs(), dy.abs()) as usize;
    lerp_closed_iter(self.ends[0], self.ends[1], n + 1)
  }

  pub fn clip(mut self) -> impl Iterator<Item = Line<T>>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    let get_x = |p: &Point3<f32>| p.x;
    let get_y = |p: &Point3<f32>| p.y;
    let get_z = |p: &Point3<f32>| p.z;

    if !self.clip_component(get_x, -1.0, 1.0) {
      return None.into_iter();
    }
    if !self.clip_component(get_y, -1.0, 1.0) {
      return None.into_iter();
    }
    if !self.clip_component(get_z, -1.0, 1.0) {
      return None.into_iter();
    }

    Some(self).into_iter()
  }

  fn clip_component<F>(&mut self, get_comp: F, min: f32, max: f32) -> bool
  where
    T: ToClipSpace + Clone + Copy + Lerp,
    F: Fn(&Point3<f32>) -> f32,
  {
    let mut av = get_comp(self.a().to_clip());
    let mut bv = get_comp(self.b().to_clip());

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
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[0] = lerp(t, self.a(), self.b());
    } else if av >= min && bv < min {
      // clip b on min
      let t = (min - av) / (bv - av);
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[1] = lerp(t, self.a(), self.b());
    }

    // recalculate because a and b may be changed
    av = get_comp(self.a().to_clip());
    bv = get_comp(self.b().to_clip());

    if av > max && bv <= max {
      // clip a on max
      let t = (max - av) / (bv - av);
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[0] = lerp(t, self.a(), self.b());
    } else if av <= max && bv > max {
      // clip b on max
      let t = (max - av) / (bv - av);
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[1] = lerp(t, self.a(), self.b());
    }

    true
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.ends.iter_mut().for_each(f)
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
  pub fn edges(&'a self) -> impl Iterator<Item = Line<T>>
  where
    T: Clone,
  {
    vec![
      (self.vertices[0].clone(), self.vertices[1].clone()).into(),
      (self.vertices[1].clone(), self.vertices[2].clone()).into(),
      (self.vertices[2].clone(), self.vertices[0].clone()).into(),
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

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.vertices.iter_mut().for_each(f)
  }

  // the caller needs to ensure self.vert.to_clip are of screen coords
  pub fn to_fill_pixels(self) -> impl Iterator<Item = T>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    let [upper, lower] = self.horizontally_split();
    let upper = upper
      .into_iter()
      .flat_map(|trig| trig.upper_trig_to_horizontal_lines());
    let lower = lower
      .into_iter()
      .flat_map(|trig| trig.lower_trig_to_horizontal_lines());
    let lines = upper.chain(lower);

    lines.flat_map(|line| line.to_horizontal_pixels())
  }

  pub fn to_edge_pixels(self) -> impl Iterator<Item = T>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    self.edges().flat_map(|line| line.to_pixels())
  }

  // the caller needs to ensure self is an upper trig and
  // self.vert.to_clip are of screen coordinates
  pub fn upper_trig_to_horizontal_lines(self) -> impl Iterator<Item = Line<T>>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    let top = self.a();
    let bottom_left = self.b();
    let bottom_right = self.c();

    let top_y = top.to_clip().y;
    let bottom_y = bottom_left.to_clip().y;
    let h = (top_y - bottom_y).abs() as usize;

    let left_pts_iter = lerp_closed_iter(*top, *bottom_left, h + 1);
    let right_pts_iter = lerp_closed_iter(*top, *bottom_right, h + 1);

    left_pts_iter.zip(right_pts_iter).map(Line::from)
  }

  pub fn lower_trig_to_horizontal_lines(self) -> impl Iterator<Item = Line<T>>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    let top_left = self.a();
    let top_right = self.b();
    let bottom = self.c();

    let top_y = top_left.to_clip().y;
    let bottom_y = bottom.to_clip().y;
    let h = (top_y - bottom_y).abs() as usize;

    let left_pts_iter = lerp_closed_iter(*top_left, *bottom, h + 1);
    let right_pts_iter = lerp_closed_iter(*top_right, *bottom, h + 1);

    left_pts_iter.zip(right_pts_iter).map(Line::from)
  }

  // returns two triangles whose vertices are ordered by y values
  pub fn horizontally_split(mut self) -> [Option<Trig<T>>; 2]
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    const EPS: f32 = 0.1;
    self
      .vertices
      .sort_unstable_by(|p1, p2| f32_cmp(&p1.to_clip().y, &p2.to_clip().y));
    let ay = self.a().to_clip().y;
    let by = self.b().to_clip().y;
    let cy = self.c().to_clip().y;

    if abs_diff_eq!(ay, cy, epsilon = EPS) {
      // a flat line
      return [Some(self), None];
    }

    if abs_diff_eq!(ay, by, epsilon = EPS) {
      // a lower triangle
      return [None, Some(self)];
    }

    if abs_diff_eq!(by, cy, epsilon = EPS) {
      // a upper triangle
      return [Some(self), None];
    }

    // a normal triangle that we need to split
    let t = (by - ay) / (cy - ay);
    // dbg!(cy - ay);
    let ptl = lerp(t, self.a(), self.c());
    let ptr = *self.b();

    let upper_trig = [*self.a(), ptl, ptr].into();
    let lower_trig = [ptl, ptr, *self.c()].into();

    [Some(upper_trig), Some(lower_trig)]
  }

  pub fn clip(self) -> SmallVec<[Trig<T>; 2]>
  where
    T: ToClipSpace + Lerp + Clone + Copy,
  {
    if self.fully_visible() {
      return smallvec![self];
    }

    let init: SmallVec<[&Trig<T>; 2]> = smallvec![&self];

    init
      .into_iter()
      .flat_map(|t| t.clip_component(|p| p.to_clip().z, -1.0, -1.0))
      .flat_map(|t| t.clip_component(|p| p.to_clip().z, 1.0, 1.0))
      .flat_map(|t| t.clip_component(|p| p.to_clip().x, -1.0, -1.0))
      .flat_map(|t| t.clip_component(|p| p.to_clip().x, 1.0, 1.0))
      .flat_map(|t| t.clip_component(|p| p.to_clip().y, -1.0, -1.0))
      .flat_map(|t| t.clip_component(|p| p.to_clip().y, 1.0, 1.0))
      .collect()
  }

  fn clip_component<F>(
    &self,
    get_comp: F,
    lim: f32,
    sign: f32,
  ) -> SmallVec<[Trig<T>; 2]>
  where
    T: ToClipSpace + Lerp + Clone + Copy,
    F: Fn(&T) -> f32,
  {
    let (va, vb, vc) = (self.a(), self.b(), self.c());
    let v: [f32; 3] = self.as_ref().map(get_comp).vertices();
    let (a, b, c) = (v[0], v[1], v[2]);

    // redefine lt, gt, le, ge operator based on the sign
    let lt = |x: f32, y: f32| x * sign < y * sign;
    let ge = |x: f32, y: f32| x * sign >= y * sign;
    let in_lim = |x: f32| lt(x, lim);
    let out_lim = |x: f32| ge(x, lim);

    // case 1: all vertex within range
    if in_lim(a) && in_lim(b) && in_lim(c) {
      return smallvec![self.clone()];
    }

    // case 2: all vertex out of range
    if out_lim(a) && out_lim(b) && out_lim(c) {
      // within the range; draw without clipping
      return smallvec![];
    }

    // case 3: two vertices out of range
    if in_lim(a) && out_lim(b) && out_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return smallvec![[*va, new_vb, new_vc].into()];
    }
    if out_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return smallvec![[new_va, *vb, new_vc].into()];
    }
    if out_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return smallvec![[new_va, new_vb, *vc].into()];
    }

    // case 4: one vertex out of range
    if out_lim(a) && in_lim(b) && in_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return smallvec![
        [*vb, *vc, new_vb].into(),
        [*vc, new_vc, new_vb].into()
      ];
    }
    if in_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return smallvec![
        [*va, *vc, new_va].into(),
        [*vc, new_va, new_vc].into()
      ];
    }
    if in_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return smallvec![
        [*va, *vb, new_va].into(),
        [*vb, new_va, new_vb].into()
      ];
    }

    unreachable!()
  }

  pub fn fully_visible(&self) -> bool
  where
    T: ToClipSpace,
  {
    let comp_in_range = |v| (-1.0..=1.0).contains(&v);
    let pt_in_range = |p: &Point3<f32>| {
      comp_in_range(p.x) && comp_in_range(p.y) && comp_in_range(p.z)
    };

    pt_in_range(self.a().to_clip())
      && pt_in_range(self.b().to_clip())
      && pt_in_range(self.c().to_clip())
  }
}

/// Types that represents a point in clip space
pub trait ToClipSpace {
  fn to_clip(&self) -> &Point3<f32>;
}

impl ToClipSpace for Pt {
  fn to_clip(&self) -> &Point3<f32> {
    &self.clip_pos
  }
}

impl ToClipSpace for Point3<f32> {
  fn to_clip(&self) -> &Point3<f32> {
    &self
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
    self.map(Into::into)
  }

  pub fn edges(&self) -> impl Iterator<Item = Line<T>> + '_
  where
    T: Copy,
  {
    let n = self.vertices.len();
    (0..n).map(move |i| {
      let a = self.vertices[i];
      let b = self.vertices[(i + 1) % n];
      Line::from((a, b))
    })
  }

  pub fn normal(&self) -> Vector3<f32>
  where
    T: ToClipSpace,
  {
    debug_assert!(self.vertices.len() >= 3);
    let v1 = self.vertices()[1].to_clip() - self.vertices()[0].to_clip();
    let v2 = self.vertices()[2].to_clip() - self.vertices()[0].to_clip();
    v1.cross(&v2)
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.vertices.iter_mut().for_each(f)
  }

  pub fn map<F, S>(self, f: F) -> Face<S>
  where
    F: Fn(T) -> S,
  {
    Face {
      vertices: self.vertices.into_iter().map(f).collect(),
      double_faced: self.double_faced,
    }
  }
}

impl<T, const N: usize> From<[T; N]> for Face<T> {
  fn from(verts: [T; N]) -> Self {
    Self {
      vertices: Vec::from(verts),
      double_faced: false,
    }
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

#[derive(Debug, Clone, Copy)]
pub struct PolyVert<'a> {
  pub vertex: &'a Point3<f32>,
  pub texture_coords: Option<&'a Vector2<f32>>,
  pub normal: Option<&'a Vector3<f32>>,
}

#[derive(Debug, Clone)]
pub struct WorldMesh<'a> {
  pub transform: Matrix4<f32>,
  pub double_faced: bool,
  pub casts_shadow: bool,
  pub mesh: Cow<'a, Mesh>,
}

impl<'a> From<&'a Mesh> for WorldMesh<'a> {
  fn from(mesh: &'a Mesh) -> Self {
    Self {
      transform: Matrix4::identity(),
      mesh: Cow::Borrowed(mesh),
      double_faced: false,
      casts_shadow: false,
    }
  }
}

impl<'a> WorldMesh<'a> {
  #[allow(unused)]
  pub fn new() -> Self {
    Self {
      transform: Matrix4::identity(),
      mesh: Default::default(),
      double_faced: false,
      casts_shadow: false,
    }
  }

  pub fn double_faced(mut self, double_faced: bool) -> Self {
    self.double_faced = double_faced;
    self
  }

  pub fn casts_shadow(mut self, casts_shadow: bool) -> Self {
    self.casts_shadow = casts_shadow;
    self
  }

  fn mesh_faces(&self) -> impl Iterator<Item = Face<PolyVert<'_>>> {
    self.mesh.faces.iter().map(move |f| self.get_face(f))
  }

  // Return faces in world world coordinates
  pub fn faces(&self) -> impl Iterator<Item = Face<Pt>> + '_ {
    self
      .mesh_faces()
      .map(move |face| face.map(|vert| Pt::from(&vert)))
  }

  pub fn get_face(&self, face: &Face<IndexedPolyVert>) -> Face<PolyVert<'_>> {
    let mut res = Face::new(self.double_faced);
    let mesh = &self.mesh;
    for vert in face.vertices() {
      let vertex = &mesh.vertices[vert.vertex_index];
      let texture_coords = vert.texture_index.map(|i| &mesh.texture_coords[i]);
      let normal = vert.normal_index.map(|i| &mesh.vertex_normals[i]);

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
    self
      .mesh
      .material
      .as_ref()
      .unwrap_or_else(|| SimpleMaterial::plaster())
  }

  pub fn apply_transformation(&self) -> Self {
    let mesh = Cow::Owned(self.mesh.apply_transformation(&self.transform));
    Self {
      mesh,
      transform: Matrix4::identity(),
      double_faced: self.double_faced,
      casts_shadow: self.casts_shadow,
    }
  }
}

pub struct Scene<'a> {
  textures: Cow<'a, TextureStash>,
  camera: Camera,
  lights: Vec<Light>,
  meshes: Vec<WorldMesh<'a>>,
}

impl<'a> Scene<'a> {
  pub fn new(camera: Camera) -> Self {
    Self {
      camera,
      textures: Cow::Owned(TextureStash::new()),
      lights: vec![],
      meshes: vec![],
    }
  }

  pub fn set_texture_stash(&mut self, textures: &'a TextureStash) {
    self.textures = Cow::Borrowed(textures);
  }

  pub fn texture_stash(&self) -> &TextureStash {
    &self.textures
  }

  pub fn add_light(&mut self, light: Light) {
    self.lights.push(light);
  }

  pub fn add_mesh(&mut self, mesh: WorldMesh<'a>) {
    self.meshes.push(mesh);
  }

  #[allow(unused)]
  pub fn add_meshes(&mut self, mesh: &[WorldMesh<'a>]) {
    self.meshes.extend_from_slice(mesh);
  }

  pub fn iter_meshes(&self) -> impl Iterator<Item = &WorldMesh> + '_ {
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
  pub render_time: f32,
}

// the coords of volume are all in clip space
pub struct ShadowVolume {
  volume: Vec<Face<Point3<f32>>>,
  shadow_distance: f32,
}

impl ShadowVolume {
  pub fn new() {
    let shadow_distance = 10000.0;
  }

  pub fn add_face(&mut self, face: &Face<Pt>, camera: &Camera, light: &Light) {
    for line in face.edges() {
      let p1 = line.a();
      let p2 = line.b();
      let p1_far = light.project(&p1.world_pos, self.shadow_distance);
      let p2_far = light.project(&p2.world_pos, self.shadow_distance);

      let face: [Point3<f32>; 4] = [
        camera.project_point(&p1.world_pos),
        camera.project_point(&p2.world_pos),
        camera.project_point(&p2_far),
        camera.project_point(&p1_far),
      ];
      self.volume.push(Face::from(face));
    }
  }

  pub fn add_world_mesh(
    &mut self,
    mesh: &WorldMesh,
    camera: &Camera,
    light: &Light,
  ) {
    for face in mesh.faces() {
      self.add_face(&face, camera, light);
    }
  }

  pub fn faces(&self) -> impl Iterator<Item = &Face<Point3<f32>>> + '_ {
    self.volume.iter()
  }
}

pub struct ShadowRasterizer<'a> {
  size: (f32, f32),
  zbuffer: &'a Image<f32>,
  stencil_buffer: Image<i32>,
}

impl<'a> ShadowRasterizer<'a> {
  pub fn new(zbuffer: &'a Image<f32>) -> Self {
    let size = zbuffer.dimension;
    let stencil_buffer = Image::new_filled(size, &0);

    let size = (size.0 as f32, size.1 as f32);

    ShadowRasterizer {
      size,
      zbuffer,
      stencil_buffer,
    }
  }

  #[allow(unused)]
  pub fn render_stencil(&mut self, volume: &ShadowVolume) {
    for face in volume.faces() {
      let sign = if Self::is_hidden_face(&face) { -1 } else { 1 };
      for trig in face.triangulate() {
        self.fill_triangle(trig, sign)
      }
    }
  }

  fn is_hidden_face(face: &Face<Point3<f32>>) -> bool {
    let positive_direction: Vector3<f32> = [0.0, 0.0, 1.0].into();
    face.normal().dot(&positive_direction) < 0.0
  }

  fn fill_triangle(&mut self, trig: Trig<Point3<f32>>, sign: i32) {
    for mut trig in trig.clip() {
      trig.map_in_place(|pt| self.to_screen_pt(pt));
      for pt in trig.to_fill_pixels() {
        let coords = (pt.x as i32, pt.y as i32);
        let pixel = self.stencil_buffer.pixel_mut(coords).unwrap();
        *pixel += sign;
      }
    }
  }

  fn to_screen_pt(&self, clip_pt: &mut Point3<f32>) {
    let (w, h) = self.size;
    clip_pt.x = 0.5 * (clip_pt.x + 1.0) * w;
    clip_pt.y = 0.5 * (1.0 - clip_pt.y) * h;
  }
}

pub struct Rasterizer {
  size: (f32, f32),
  mode: RasterizerMode,
  image: Image<Color>,
  zbuffer: Image<f32>,
  metric: RasterizerMetric,
  shader_options: ShaderOptions,
}

impl Rasterizer {
  pub fn new(size: (usize, usize)) -> Self {
    let image = Image::new(size);
    let zbuffer = Image::new_filled(size, &1.01);
    let mode = RasterizerMode::Shaded;
    let size = (image.width() as f32, image.height() as f32);
    let metric = Default::default();
    let shader_options = Default::default();

    Self {
      size,
      image,
      zbuffer,
      mode,
      metric,
      shader_options,
    }
  }

  pub fn set_shader_options(&mut self, options: ShaderOptions) {
    self.shader_options = options;
  }

  pub fn set_mode(&mut self, mode: RasterizerMode) {
    self.mode = mode;
  }

  pub fn rasterize(&mut self, scene: &Scene) {
    let now = Instant::now();

    for mesh in scene.iter_meshes() {
      let mesh = mesh.apply_transformation();
      for mut face in mesh.faces() {
        let context = self.shader_context(scene, &mesh);
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

    self.metric.render_time = now.elapsed().as_secs_f32();
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
          self.fill_triangle(trig, &context, shader);
        }
      }
      Clipped => {
        for trig in face.triangulate() {
          self.draw_triangle_clipped(trig, &context, shader);
        }
      }
      Wireframe => {
        for line in face.edges() {
          self.draw_line(line, &context, shader);
        }
      }
    }
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
    let coords = (p.clip_pos.x as i32, p.clip_pos.y as i32);
    let (x, y) = coords;
    let (w, h) = self.size();
    if x < 0 || x >= w || y < 0 || y >= h {
      return;
    }

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
    face.normal().dot(&positive_direction) < 0.0
  }

  // do not check for zbuffer
  fn put_pixel(&mut self, coords: (i32, i32), color: Color) {
    if let Some(pixel) = self.image.pixel_mut(coords) {
      *pixel = color;
    }
  }

  fn draw_line(
    &mut self,
    line: Line<Pt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    for mut line in line.clip() {
      line.map_in_place(|pt| self.to_screen_pt(pt));

      for pt in line.to_pixels() {
        self.draw_pixel(pt, context, shader);
      }
    }
  }

  // fill a triangle that is flat at bottom
  fn shader_context<'a>(
    &self,
    scene: &'a Scene,
    mesh: &WorldMesh,
  ) -> ShaderContext<'a> {
    ShaderContext {
      textures: scene.texture_stash(),
      camera: scene.camera().matrix(),
      model: mesh.transform.clone(),
      lights: scene.lights(),
      options: self.shader_options.clone(),
    }
  }

  fn draw_triangle_clipped(
    &mut self,
    trig: Trig<Pt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    for mut trig in trig.clip() {
      trig.map_in_place(|pt| self.to_screen_pt(pt));

      for pt in trig.to_edge_pixels() {
        self.draw_pixel(pt, context, shader);
      }
    }
  }

  fn fill_triangle(
    &mut self,
    trig: Trig<Pt>,
    context: &ShaderContext,
    shader: &dyn Shader,
  ) {
    self.metric.triangles_rendered += 1;

    for mut trig in trig.clip() {
      trig.map_in_place(|pt| self.to_screen_pt(pt));

      for pt in trig.to_fill_pixels() {
        self.draw_pixel(pt, context, shader);
      }
    }
  }

  fn to_screen_pt(&self, pt: &mut Pt) {
    let (w, h) = self.size_f32();
    let p = &mut pt.clip_pos;
    p.x = 0.5 * w * (p.x + 1.0);
    p.y = 0.5 * h * (1.0 - p.y);
  }

  fn size(&self) -> (i32, i32) {
    (self.size.0 as i32, self.size.1 as i32)
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
