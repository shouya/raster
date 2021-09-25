use std::{collections::HashMap, fs::File, io::Read, path::Path, str::FromStr};

use anyhow::{ensure, Result};
use nalgebra::{Point3, Vector2, Vector3};

use crate::{
  raster::{Color, Face, Image, IndexedPolyVert, COLOR},
  shader::{SimpleMaterial, TextureStash},
};

struct Mtl {
  pub map: HashMap<String, SimpleMaterial>,
  pub textures: TextureStash,
}

impl Mtl {
  pub fn new() -> Self {
    Self {
      map: HashMap::new(),
      textures: TextureStash::new(),
    }
  }

  pub fn load(path: &Path) -> Result<Self> {
    let mut buf = String::new();
    File::open(path)?.read_to_string(&mut buf)?;
    Self::parse(&buf, path.parent().unwrap())
  }

  pub fn parse(s: &str, rel_path: &Path) -> Result<Self> {
    let mut textures = TextureStash::new();
    let mut res = HashMap::new();
    let mut curr_mtl_name = None;
    let mut curr_mat = SimpleMaterial::default();

    for line in s.lines() {
      if line.starts_with("#") {
        continue;
      }
      if line.trim().is_empty() {
        continue;
      }

      let segments = line.split(" ").collect::<Vec<_>>();
      match &segments[..] {
        ["newmtl", name] => match curr_mtl_name {
          Some(old_name) => {
            res.insert(old_name, curr_mat);
            curr_mat = Default::default();
            curr_mtl_name = Some((*name).into());
          }
          None => {
            curr_mtl_name = Some((*name).into());
          }
        },
        ["Ka", color @ ..] => curr_mat.ambient_color = parse_color(color)?,
        ["Kd", color @ ..] => curr_mat.diffuse_color = parse_color(color)?,
        ["Ks", color @ ..] => curr_mat.specular_color = parse_color(color)?,
        ["Ns", v] => curr_mat.specular_highlight = parse_float(v)?,
        ["d", d] => curr_mat.dissolve = parse_float(d)?,
        ["map_Kd", texture @ ..] => {
          let handle = textures.add(parse_texture_color(texture, rel_path)?);
          curr_mat.color_texture = Some(handle);
        }
        [_any @ ..] => {}
      }
    }

    if let Some(name) = curr_mtl_name {
      res.insert(name, curr_mat);
    }

    Ok(Mtl { map: res, textures })
  }

  fn get(&self, material_name: &str) -> Result<SimpleMaterial> {
    let mat = self
      .map
      .get(material_name)
      .ok_or(anyhow::anyhow!("Material {:?} not found", material_name))?
      .clone();
    Ok(mat)
  }
}

#[derive(Clone, Default, Debug)]
pub struct Mesh {
  pub material: Option<SimpleMaterial>,
  pub vertices: Vec<Point3<f32>>,
  pub vertex_normals: Vec<Vector3<f32>>,
  pub texture_coords: Vec<Vector2<f32>>,
  pub faces: Vec<Face<IndexedPolyVert>>,
}

impl Mesh {
  pub fn new() -> Self {
    Default::default()
  }

  pub fn add_simple_face(&mut self, vertices: &[Point3<f32>]) {
    let n = self.vertices.len();
    self.vertices.extend_from_slice(vertices);
    let mut face = Face::new(false);
    for i in 0..vertices.len() {
      face.add_vert(IndexedPolyVert::new(n + i));
    }
    self.faces.push(face);
  }
}

struct Obj {
  pub mtl: Mtl,
  pub objs: Vec<Mesh>,
}

impl Obj {
  fn new() -> Self {
    Self {
      mtl: Mtl::new(),
      objs: vec![],
    }
  }

  pub fn load(path: &Path) -> Result<Self> {
    let mut buf = String::new();
    File::open(path)?.read_to_string(&mut buf)?;
    Self::parse(&buf, path.parent().unwrap())
  }

  pub fn parse(s: &str, rel_path: &Path) -> Result<Self> {
    let mut obj = Obj::new();
    let mut curr_mesh = Mesh::default();

    for line in s.lines() {
      if line.starts_with("#") {
        continue;
      }
      if line.trim().is_empty() {
        continue;
      }
      let segments = line.split(" ").collect::<Vec<_>>();
      match segments.as_slice() {
        ["v", vs @ ..] => curr_mesh.vertices.push(parse_point3(vs)?),
        ["vn", vs @ ..] => curr_mesh.vertex_normals.push(parse_vec3(vs)?),
        ["vt", vs @ ..] => curr_mesh.texture_coords.push(parse_vec2(vs)?),
        ["f", fs @ ..] => curr_mesh.faces.push(parse_face(fs)?),
        ["mtllib", f] => {
          if let Ok(mtl) = Mtl::load(&rel_path.join(Path::new(f))) {
            obj.mtl = mtl;
          }
        }
        ["usemtl", m] => {
          // TODO: change to a less clumsy way
          if curr_mesh.faces.len() > 0 {
            obj.objs.push(curr_mesh.clone());
            curr_mesh.faces = vec![];
          }
          // it's okay that the material is not found
          if let Ok(mat) = obj.mtl.get(m) {
            curr_mesh.material = Some(mat);
          }
        }
        _ => {}
      }
    }

    // TODO: change to a less clumsy way
    if curr_mesh.faces.len() > 0 {
      obj.objs.push(curr_mesh);
    }

    Ok(obj)
  }
}

fn parse_floats(slices: &[&str]) -> Result<Vec<f32>> {
  slices
    .iter()
    .map(|s| parse_float(s))
    .collect::<Result<Vec<_>, _>>()
}

fn parse_face(slices: &[&str]) -> Result<Face<IndexedPolyVert>> {
  let mut face = Face::new(false);
  for v in slices {
    let indices: Vec<Result<usize, _>> =
      v.split("/").map(|i| usize::from_str(i)).collect();

    match indices.as_slice() {
      [Ok(vi)] => face.add_vert(IndexedPolyVert::new(vi - 1)),
      [Ok(vi), Ok(ti)] => {
        face.add_vert(IndexedPolyVert::new_texture(vi - 1, ti - 1))
      }
      [Ok(vi), Ok(ti), Ok(ni)] => face
        .add_vert(IndexedPolyVert::new_texture_normal(vi - 1, ti - 1, ni - 1)),
      [Ok(vi), Err(_), Ok(ni)] => {
        face.add_vert(IndexedPolyVert::new_normal(vi - 1, ni - 1))
      }
      _ => anyhow::bail!("Invalid face: {:?}", indices),
    }
  }

  Ok(face)
}

fn parse_color(color: &[&str]) -> Result<Color> {
  ensure!(
    color.len() == 3,
    "{:?} doesn't have three components",
    color
  );
  let floats = parse_floats(color)?;
  Ok(COLOR::rgb(floats[0], floats[1], floats[2]))
}

fn parse_float(s: &str) -> Result<f32> {
  Ok(f32::from_str(s)?)
}

fn parse_point3(vs: &[&str]) -> Result<Point3<f32>> {
  Ok(Point3::from_slice(&parse_floats(vs)?))
}
fn parse_vec3(vs: &[&str]) -> Result<Vector3<f32>> {
  Ok(Vector3::from_column_slice(&parse_floats(vs)?))
}
fn parse_vec2(vs: &[&str]) -> Result<Vector2<f32>> {
  Ok(Vector2::from_column_slice(&parse_floats(vs)?))
}

fn parse_texture_color(
  options: &[&str],
  rel_path: &Path,
) -> Result<Image<Color>> {
  ensure!(
    options.len() == 1,
    "texture options are not supported: {:?}",
    options
  );
  let path = rel_path.join(options[0]);
  let imgfile = image::io::Reader::open(&path)?
    .with_guessed_format()?
    .decode()?
    .into_rgb8();

  let dim = imgfile.dimensions();
  let mut img = Image::new((dim.0 as usize, dim.1 as usize));
  for (x, y, pixel) in imgfile.enumerate_pixels() {
    let p = image::Pixel::channels(pixel);
    let comp = |n| p[n] as f32 / 255.0;
    let color = COLOR::rgb(comp(0), comp(1), comp(2));

    img.put_pixel((x as i32, y as i32), color);
  }

  Ok(img)
}

pub struct MeshObject {
  pub meshes: Vec<Mesh>,
  pub textures: TextureStash,
}

pub fn load(path: &Path) -> Result<MeshObject> {
  let obj = Obj::load(path)?;
  Ok(MeshObject {
    textures: obj.mtl.textures,
    meshes: obj.objs,
  })
}
