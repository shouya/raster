use std::{collections::HashMap, rc::Rc, time::Instant};

use approx::abs_diff_eq;
use smallvec::{smallvec, SmallVec};

use crate::{
  lerp::{lerp, lerp_closed_iter, Lerp},
  shader::{
    Light, Shader, ShaderContext, ShaderOptions, SimpleMaterial, TextureStash,
  },
  types::{Mat4, Point3, Point3Ord, Vector2, Vector3, Vector4},
  util::f32_cmp,
  wavefront::Mesh,
};

pub type Color = Vector4;

#[allow(non_snake_case)]
pub(crate) mod COLOR {
  use super::*;

  pub fn rgb(r: f32, g: f32, b: f32) -> Color {
    rgba(r, g, b, 1.0)
  }

  pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
    glam::vec4(r, g, b, a)
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

  pub fn new_filled(size: (usize, usize), val: T) -> Self
  where
    T: Copy,
  {
    let len = size.0 * size.1;
    let buffer = vec![val; len];

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

  pub fn pixels_with_coords(&self) -> impl Iterator<Item = ((i32, i32), &T)> {
    let w = self.width();
    self.pixels.iter().enumerate().map(move |(i, p)| {
      let x = i % w;
      let y = i / w;
      ((x as i32, y as i32), p)
    })
  }

  pub fn pixels_mut_with_coords(
    &mut self,
  ) -> impl Iterator<Item = ((i32, i32), &mut T)> {
    let w = self.width();
    self.pixels.iter_mut().enumerate().map(move |(i, p)| {
      let x = i % w;
      let y = i / w;
      ((x as i32, y as i32), p)
    })
  }

  pub fn pixels_mut(&mut self) -> impl Iterator<Item = &mut T> {
    self.pixels.iter_mut()
  }

  pub fn as_ref(&self) -> Image<&T> {
    Image {
      dimension: self.dimension,
      pixels: self.pixels.iter().collect(),
    }
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
  pub fn nearest_color(&self, uv: Vector2) -> Color {
    let coords = self.nearest_coords(uv, (0, 0));
    *self.pixel(coords).unwrap()
  }

  pub fn bilinear_color(&self, uv: Vector2) -> Color {
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
  fn nearest_coords(&self, uv: Vector2, offset: (i32, i32)) -> (i32, i32) {
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
  inv_transform: Mat4,
  perspective: Mat4,
}

impl Camera {
  pub fn new_perspective(
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
  ) -> Self {
    let perspective = Mat4::perspective_rh_gl(fovy, aspect, znear, zfar);
    let inv_transform = Mat4::IDENTITY;
    Self {
      perspective,
      inv_transform,
    }
  }

  pub fn matrix(&self) -> Mat4 {
    self.perspective * self.inv_transform
  }

  pub fn transformd(&mut self, trans: &Mat4) {
    self.inv_transform *= trans.inverse();
  }

  pub fn project_point(&self, pt: &Point3) -> Point3 {
    self.matrix().project_point3((*pt).into()).into()
  }
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, PartialOrd, Ord)]
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

  // reorder ends
  pub fn sort_ends(&mut self)
  where
    T: Ord,
  {
    self.ends.sort()
  }

  // The caller needs to ensure "self" is in screen coordinates
  pub fn to_horizontal_pixels(self) -> impl Iterator<Item = T>
  where
    T: ToClipSpace + Clone + Copy + Lerp,
  {
    // ensure the line is indeed flat
    debug_assert!((self.a().to_clip().y - self.b().to_clip().y) as i32 == 0);
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
    let get_x = |p: &Point3| p.x;
    let get_y = |p: &Point3| p.y;
    let get_z = |p: &Point3| p.z;

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
    F: Fn(&Point3) -> f32,
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

  pub fn map<S, F>(self, f: F) -> Line<S>
  where
    F: Fn(T) -> S,
  {
    Line {
      ends: self.ends.map(|x| f(x)),
    }
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
    let pt_in_range = |p: &Point3| {
      comp_in_range(p.x) && comp_in_range(p.y) && comp_in_range(p.z)
    };

    pt_in_range(self.a().to_clip())
      && pt_in_range(self.b().to_clip())
      && pt_in_range(self.c().to_clip())
  }
}

/// Types that represents a point in clip space
pub trait ToClipSpace {
  fn to_clip(&self) -> &Point3;
  fn to_clip_mut(&mut self) -> &mut Point3;

  fn scale_to_screen(&mut self, (w, h): (f32, f32)) {
    let p = self.to_clip_mut();
    p.x = 0.5 * w * (p.x + 1.0);
    p.y = 0.5 * h * (1.0 - p.y);
  }
}

impl ToClipSpace for Pt {
  fn to_clip(&self) -> &Point3 {
    &self.clip_pos
  }
  fn to_clip_mut(&mut self) -> &mut Point3 {
    &mut self.clip_pos
  }
}

impl ToClipSpace for Point3 {
  fn to_clip(&self) -> &Point3 {
    &self
  }
  fn to_clip_mut(&mut self) -> &mut Point3 {
    self
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

  pub fn len(&self) -> usize {
    self.vertices.len()
  }

  /// the caller needs to ensure T is in clip space.
  pub fn is_hidden(&self) -> bool
  where
    T: ToClipSpace,
  {
    if self.double_faced() {
      return false;
    }

    let positive_direction: Vector3 = [0.0, 0.0, 1.0].into();
    self.normal().dot(positive_direction) < 0.0
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

  pub fn normal(&self) -> Vector3
  where
    T: ToClipSpace,
  {
    debug_assert!(self.vertices.len() >= 3);
    let v1 = *self.vertices()[1].to_clip() - *self.vertices()[0].to_clip();
    let v2 = *self.vertices()[2].to_clip() - *self.vertices()[0].to_clip();
    v1.cross(v2)
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
  pub vertex: &'a Point3,
  pub texture_coords: Option<&'a Vector2>,
  pub normal: Option<&'a Vector3>,
}

#[derive(Clone)]
pub struct WorldMesh {
  pub transform: Option<Mat4>,
  pub double_faced: bool,
  pub casts_shadow: bool,
  pub mesh: Rc<Mesh>,
}

impl From<Rc<Mesh>> for WorldMesh {
  fn from(mesh: Rc<Mesh>) -> Self {
    Self {
      mesh,
      transform: None,
      double_faced: false,
      casts_shadow: true,
    }
  }
}

impl WorldMesh {
  #[allow(unused)]
  pub fn new() -> Self {
    Self {
      transform: None,
      mesh: Rc::new(Default::default()),
      double_faced: false,
      casts_shadow: true,
    }
  }

  pub fn double_faced(mut self, double_faced: bool) -> Self {
    self.double_faced = double_faced;
    self
  }

  #[allow(unused)]
  pub fn casts_shadow(mut self, casts_shadow: bool) -> Self {
    self.casts_shadow = casts_shadow;
    self
  }

  fn mesh_faces(&self) -> impl Iterator<Item = Face<PolyVert<'_>>> {
    self.mesh.faces.iter().map(move |f| self.get_face(f))
  }

  pub fn set_shader(mut self, shader: Box<dyn Shader>) -> Self {
    Rc::make_mut(&mut self.mesh).set_material(shader);
    self
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

  pub fn transformed(mut self, transform: Mat4) -> Self {
    let current_transform = self.transform.unwrap_or(Mat4::IDENTITY);
    self.transform = Some(transform * current_transform);
    self
  }

  pub fn shader(&self) -> &dyn Shader {
    self
      .mesh
      .material
      .as_ref()
      .map(|x| x.as_ref())
      .unwrap_or_else(|| SimpleMaterial::plaster())
  }

  pub fn apply_transformation(&self) -> Self {
    match self.transform {
      Some(transform) => {
        let mesh = self.mesh.apply_transformation(&transform);
        Self {
          mesh: Rc::new(mesh),
          transform: None,
          double_faced: self.double_faced,
          casts_shadow: self.casts_shadow,
        }
      }
      None => self.clone(),
    }
  }
}

pub struct Scene {
  textures: Rc<TextureStash>,
  camera: Camera,
  lights: Vec<Light>,
  meshes: Vec<WorldMesh>,
}

impl Scene {
  pub fn new(camera: Camera) -> Self {
    Self {
      camera,
      textures: Rc::new(TextureStash::new()),
      lights: vec![],
      meshes: vec![],
    }
  }

  pub fn set_texture_stash(&mut self, textures: Rc<TextureStash>) {
    self.textures = textures;
  }

  pub fn texture_stash(&self) -> &TextureStash {
    &self.textures
  }

  pub fn add_light(&mut self, light: Light) {
    self.lights.push(light);
  }

  pub fn add_mesh(&mut self, mesh: WorldMesh) {
    self.meshes.push(mesh);
  }

  pub fn visualize_light(&mut self, mesh: Rc<Mesh>) {
    let lights = &self.lights;
    let meshes = &mut self.meshes;

    for light in lights {
      meshes.push(light.to_world_mesh(mesh.clone()).clone());
    }
  }

  pub fn iter_meshes(&self) -> impl Iterator<Item = &WorldMesh> + '_ {
    self.meshes.iter().map(|x| x)
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
  pub clip_pos: Point3,
  pub world_pos: Point3,
  pub color: Color,
  pub normal: Vector3,
  pub uv: Vector2,
  pub in_shadow: Option<bool>,
  pub buf_v2: Option<Vector2>,
  pub buf_v3: Option<Vector3>,
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
  pub fn new(point: Point3) -> Self {
    Self {
      clip_pos: point,
      world_pos: point,
      color: COLOR::rgba(1.0, 0.0, 0.0, 1.0),
      uv: Vector2::new(0.0, 0.0),
      normal: Vector3::ZERO,
      in_shadow: None,
      buf_v2: None,
      buf_v3: None,
    }
  }

  pub fn set_uv(&mut self, uv: Vector2) {
    self.uv = uv;
  }

  pub fn set_normal(&mut self, normal: Vector3) {
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
      in_shadow: None,
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
  volume: Vec<Face<Point3>>,
  // key absent: new edge
  // >0: edge is light facing
  // <0: edge is light away
  // =0: edge is silhouette
  edge_index: HashMap<Line<Point3Ord>, i32>,
  shadow_distance: f32,
}

#[allow(unused)]
impl ShadowVolume {
  pub fn new() -> Self {
    Self {
      volume: vec![],
      edge_index: HashMap::new(),
      shadow_distance: 1000.0,
    }
  }

  pub fn face_count(&self) -> usize {
    self.volume.len()
  }

  pub fn add_face(&mut self, face: &Face<Pt>, camera: &Camera, light: &Light) {
    let sign = if face.normal().dot(light.pos().normalize()) < 0.0 {
      -1
    } else {
      1
    };

    for line in face.edges() {
      let key = self.register_edge(&line, sign);
      if !self.is_silhouette_edge(&key) {
        continue;
      }

      let p1 = line.a().world_pos;
      let p2 = line.b().world_pos;

      let p1_near = light.project(&p1, 0.0001);
      let p2_near = light.project(&p2, 0.0001);
      let p1_far = light.project(&p1, self.shadow_distance);
      let p2_far = light.project(&p2, self.shadow_distance);

      let face: Face<Point3> = [
        camera.project_point(&p1_near),
        camera.project_point(&p2_near),
        camera.project_point(&p2_far),
        camera.project_point(&p1_far),
      ]
      .into();

      self.volume.push(Face::from(face));
      self.deregister_edge(&key);
    }
  }

  pub fn add_world_mesh(
    &mut self,
    mesh: &WorldMesh,
    camera: &Camera,
    light: &Light,
  ) {
    if !mesh.casts_shadow {
      return;
    }

    for face in mesh.faces() {
      self.add_face(&face, camera, light);
    }
  }

  pub fn faces(&self) -> impl Iterator<Item = &Face<Point3>> + '_ {
    self.volume.iter()
  }

  fn register_edge(&mut self, line: &Line<Pt>, sign: i32) -> Line<Point3Ord> {
    let mut key = line.clone().map(|x| Point3Ord::new(x.world_pos));
    key.sort_ends();

    let val = self.edge_index.entry(key.clone()).or_insert(0);
    *val += sign;

    key
  }

  fn is_silhouette_edge(&self, key: &Line<Point3Ord>) -> bool {
    let v = self.edge_index.get(key).unwrap_or(&0);
    *v == 0
  }

  fn deregister_edge(&mut self, key: &Line<Point3Ord>) {
    self.edge_index.remove(key);
  }
}

pub struct Rasterizer<'a> {
  size: (f32, f32),
  mode: RasterizerMode,
  // smaller value = closer to camera; range: [-1, 1]
  zbuffer: Image<Option<f32>>,
  image: Image<Option<Pt>>,
  pixel_shader: Image<Option<&'a dyn Shader>>,
  metric: RasterizerMetric,
  shader_options: ShaderOptions,
  render_shadow: bool,
  stencil_buffer: Option<Image<i32>>,
}

impl<'a> Rasterizer<'a> {
  pub fn new(size: (usize, usize)) -> Self {
    let image = Image::new_filled(size, None);
    let pixel_shader = Image::new_filled(size, None);
    let zbuffer = Image::new_filled(size, None);
    let mode = RasterizerMode::Shaded;
    let size = (image.width() as f32, image.height() as f32);
    let metric = Default::default();
    let shader_options = Default::default();

    Self {
      size,
      zbuffer,
      image,
      pixel_shader,
      mode,
      metric,
      shader_options,
      stencil_buffer: None,
      render_shadow: false,
    }
  }

  pub fn set_shader_options(&mut self, options: ShaderOptions) {
    self.shader_options = options;
  }

  pub fn set_mode(&mut self, mode: RasterizerMode) {
    self.mode = mode;
  }

  pub fn set_render_shadow(&mut self, render_shadow: bool) {
    self.render_shadow = render_shadow;
  }

  pub fn rasterize(&mut self, scene: &'a Scene) {
    let now = Instant::now();

    if self.render_shadow {
      let mut shadow_volume = ShadowVolume::new();
      self.rasterize_pixels_with_shadow(scene, &mut shadow_volume);
      self.rasterize_shadow_volume(&shadow_volume);
      self.save_shadow_info_in_pixels();
    } else {
      self.rasterize_pixels(scene);
    }

    self.shade_pixels(scene);

    self.metric.render_time = now.elapsed().as_secs_f32();
  }

  #[inline(never)]
  pub fn rasterize_pixels_with_shadow(
    &mut self,
    scene: &'a Scene,
    volume: &mut ShadowVolume,
  ) {
    let context = self.shader_context(scene);

    debug_assert!(scene.lights().len() == 1);
    let light = scene.lights().first().unwrap();
    let camera = scene.camera();

    for mesh in scene.iter_meshes() {
      let shader = mesh.shader();
      let mesh = mesh.apply_transformation();

      volume.add_world_mesh(&mesh, camera, light);

      for mut face in mesh.faces() {
        face.map_in_place(|mut p| shader.vertex(&context, &mut p));
        self.metric.vertices_shaded += face.len();

        if face.is_hidden() {
          self.metric.hidden_face_removed += 1;
          continue;
        }

        self.metric.faces_rendered += 1;
        self.rasterize_face(&mut face, shader);
      }
    }
  }

  #[inline(never)]
  pub fn rasterize_shadow_volume(&mut self, volume: &ShadowVolume) {
    let size = (self.size.0 as usize, self.size.1 as usize);
    let mut stencil_buffer = Image::new_filled(size, 0);

    for face in volume.faces() {
      let sign = if face.is_hidden() { 1 } else { -1 };
      for trig in face.triangulate() {
        self.fill_trig_in_stencil_buffer(trig, sign, &mut stencil_buffer);
      }
    }

    self.stencil_buffer = Some(stencil_buffer);
  }

  #[inline(never)]
  fn fill_trig_in_stencil_buffer(
    &self,
    trig: Trig<Point3>,
    sign: i32,
    buffer: &mut Image<i32>,
  ) {
    let (w, h) = self.size();

    for mut trig in trig.clip() {
      trig.map_in_place(|pt| pt.scale_to_screen(self.size_f32()));

      for pt in trig.to_fill_pixels() {
        let (x, y) = (pt.x as i32, pt.y as i32);

        if x < 0 || x >= w || y < 0 || y >= h {
          return;
        }

        let coords = (x, y);
        if let Some(depth) = *self.zbuffer.pixel(coords).unwrap() {
          if pt.z < depth {
            let pixel = buffer.pixel_mut(coords).unwrap();
            *pixel += sign;
          }
        }
      }
    }
  }

  fn save_shadow_info_in_pixels(&mut self) {
    let stencil_buffer = self.stencil_buffer.as_ref().unwrap();

    for (coords, pixel) in self.image.pixels_mut_with_coords() {
      if let Some(pt) = pixel {
        let stencil_value = stencil_buffer.pixel(coords).unwrap();
        pt.in_shadow = Some(*stencil_value != 0);
      }
    }
  }

  pub fn rasterize_pixels(&mut self, scene: &'a Scene) {
    let context = self.shader_context(scene);

    for mesh in scene.iter_meshes() {
      let shader = mesh.shader();
      let mesh = mesh.apply_transformation();

      for mut face in mesh.faces() {
        face.map_in_place(|mut p| shader.vertex(&context, &mut p));
        self.metric.vertices_shaded += face.len();

        if face.is_hidden() {
          self.metric.hidden_face_removed += 1;
          continue;
        }

        self.metric.faces_rendered += 1;
        self.rasterize_face(&mut face, shader);
      }
    }
  }

  pub fn rasterize_face(
    &mut self,
    face: &mut Face<Pt>,
    shader: &'a dyn Shader,
  ) {
    use RasterizerMode::*;

    match self.mode {
      Shaded => {
        for trig in face.triangulate() {
          self.fill_triangle(trig, shader);
        }
      }
      Clipped => {
        for trig in face.triangulate() {
          self.draw_triangle_clipped(trig, shader);
        }
      }
      Wireframe => {
        for line in face.edges() {
          self.draw_line(line, shader);
        }
      }
    }
  }

  fn shade_pixels(&mut self, scene: &Scene) {
    let context = self.shader_context(scene);
    let pixels = self.image.pixels_mut().zip(self.pixel_shader.pixels());
    for pixel in pixels {
      match pixel {
        (None, None) => (),
        (Some(pt), Some(shader)) => Self::shade_pixel(pt, &context, *shader),
        _ => unreachable!(),
      }
    }
  }

  pub fn zbuffer_image(&self) -> Image<Color> {
    let to_comp = |d: Option<f32>| (d.unwrap_or(2.0) + 1.0) / 2.0;
    let to_color = |d| COLOR::rgb(to_comp(d), to_comp(d), to_comp(d));
    self.zbuffer.clone().map(to_color)
  }

  pub fn stencil_buffer_image(&self) -> Image<Color> {
    let size = self.size_usize();
    let default_image = Image::new_filled(size, 0);
    let image = self.stencil_buffer.as_ref().unwrap_or(&default_image);
    let to_color = |v: &i32| {
      let c = if *v < 0 {
        1.0
      } else if *v > 0 {
        0.0
      } else {
        0.5
      };
      COLOR::rgb(c, c, c)
    };
    image.as_ref().map(to_color)
  }

  pub fn metric(&self) -> RasterizerMetric {
    self.metric.clone()
  }

  pub fn into_image(self) -> Image<Color> {
    let to_color =
      |p: Option<Pt>| p.map(|x| x.color).unwrap_or(COLOR::rgb(0.0, 0.0, 0.0));
    self.image.map(to_color)
  }

  pub fn size_f32(&self) -> (f32, f32) {
    self.size
  }

  fn shade_pixel(pt: &mut Pt, context: &ShaderContext, shader: &dyn Shader) {
    // TODO: make metric modifiable without mutability
    // self.metric.pixels_shaded += 1;
    shader.fragment(context, pt)
  }

  // only put pixel in image buffer. does not shade it with color yet
  fn rasterize_pixel(&mut self, pt: Pt, shader: &'a dyn Shader) {
    let (x, y) = (pt.clip_pos.x as i32, pt.clip_pos.y as i32);
    let (w, h) = self.size();

    if x < 0 || x >= w || y < 0 || y >= h {
      return;
    }

    let coords = (x, y);
    match self.zbuffer.pixel(coords).unwrap() {
      Some(d) if pt.depth() >= *d => {
        self.metric.pixels_discarded += 1;
        return;
      }
      None | Some(_) => {
        self.metric.pixels_shaded += 1;

        self.image.put_pixel(coords, Some(pt));
        self.pixel_shader.put_pixel(coords, Some(shader));

        self.zbuffer.put_pixel(coords, Some(pt.depth()));
      }
    }
  }

  fn draw_line(&mut self, line: Line<Pt>, shader: &'a dyn Shader) {
    for mut line in line.clip() {
      line.map_in_place(|pt| pt.scale_to_screen(self.size_f32()));

      for pt in line.to_pixels() {
        self.rasterize_pixel(pt, shader);
      }
    }
  }

  // fill a triangle that is flat at bottom
  fn shader_context<'b>(&self, scene: &'b Scene) -> ShaderContext<'b> {
    ShaderContext {
      textures: scene.texture_stash(),
      camera: scene.camera().matrix(),
      lights: scene.lights(),
      options: self.shader_options.clone(),
    }
  }

  fn draw_triangle_clipped(&mut self, trig: Trig<Pt>, shader: &'a dyn Shader) {
    for mut trig in trig.clip() {
      trig.map_in_place(|pt| pt.scale_to_screen(self.size_f32()));

      for pt in trig.to_edge_pixels() {
        self.rasterize_pixel(pt, shader);
      }
    }
  }

  fn fill_triangle(&mut self, trig: Trig<Pt>, shader: &'a dyn Shader) {
    self.metric.triangles_rendered += 1;

    for mut trig in trig.clip() {
      trig.map_in_place(|pt| pt.scale_to_screen(self.size_f32()));

      for pt in trig.to_fill_pixels() {
        self.rasterize_pixel(pt, shader);
      }
    }
  }

  fn size(&self) -> (i32, i32) {
    (self.size.0 as i32, self.size.1 as i32)
  }

  fn size_usize(&self) -> (usize, usize) {
    (self.size.0 as usize, self.size.1 as usize)
  }
}

#[cfg(test)]
mod test {
  use std::f32::consts::PI;

  use super::*;

  #[test]
  fn test_camera() {
    assert!(false);
  }
}
