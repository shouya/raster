use std::{
  collections::HashMap, fs::File, io::Read, path::Path, rc::Rc, str::FromStr,
};

use anyhow::{bail, ensure, Result};

use crate::{
  mesh::Mesh,
  raster::{Color, Image, COLOR},
  shader::{SimpleMaterial, TextureStash},
  types::{Vec2, Vec3},
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
          if let Ok(texture) = parse_texture(texture, rel_path) {
            let handle = textures.add(texture);
            curr_mat.color_texture = Some(handle);
          }
        }
        ["map_Bump", texture @ ..] => {
          if let Ok(texture) = parse_texture(texture, rel_path) {
            let handle = textures.add(texture);
            curr_mat.bump_texture = Some(handle);
          }
        }
        [_any @ ..] => {
          // unrecognized options, skipping
        }
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
    let mut curr_mesh = Mesh::new();
    let mut v = Vec::new();
    let mut vt = Vec::new();
    let mut vn = Vec::new();

    for line in s.lines() {
      if line.starts_with("#") {
        continue;
      }
      if line.trim().is_empty() {
        continue;
      }
      let segments = line.split(" ").collect::<Vec<_>>();
      match segments.as_slice() {
        ["v", vs @ ..] => v.push(parse_point3(vs)?),
        ["vt", vs @ ..] => vt.push(parse_vec2(vs)?),
        ["vn", vs @ ..] => vn.push(parse_vec3(vs)?),
        ["f", fs @ ..] => {
          let index = parse_face(fs)?;
          let polyverts = Self::resolve_verts(index, &v, &vt, &vn);
          curr_mesh.add_face(&polyverts);
        }
        ["mtllib", f] => {
          if let Ok(mtl) = Mtl::load(&rel_path.join(Path::new(f))) {
            obj.mtl = mtl;
          }
        }
        ["usemtl", m] => {
          // TODO: change to a less clumsy way
          if curr_mesh.faces.len() > 0 {
            curr_mesh.seal();
            obj.objs.push(curr_mesh.clone());
            curr_mesh = Mesh::new();
          }
          // it's okay that the material is not found
          if let Ok(mat) = obj.mtl.get(m) {
            curr_mesh.material = Some(Rc::new(mat));
          }
        }
        _ => {}
      }
    }

    // TODO: change to a less clumsy way
    if curr_mesh.faces.len() > 0 {
      curr_mesh.seal();
      obj.objs.push(curr_mesh);
    }

    Ok(obj)
  }

  fn resolve_verts(
    index: Vec<(usize, Option<usize>, Option<usize>)>,
    v: &Vec<Vec3>,
    vt: &Vec<Vec2>,
    vn: &Vec<Vec3>,
  ) -> Vec<(Vec3, Option<Vec2>, Option<Vec3>)> {
    index
      .into_iter()
      .map(|(vi, vti, vni)| (v[vi], vti.map(|i| vt[i]), vni.map(|i| vn[i])))
      .collect()
  }
}

fn parse_floats(slices: &[&str]) -> Result<Vec<f32>> {
  slices
    .iter()
    .map(|s| parse_float(s))
    .collect::<Result<Vec<_>, _>>()
}

fn parse_face(
  slices: &[&str],
) -> Result<Vec<(usize, Option<usize>, Option<usize>)>> {
  let mut res = Vec::new();
  for v in slices {
    let indices: Vec<Result<usize, _>> =
      v.split("/").map(|i| usize::from_str(i)).collect();

    let vert = match indices.as_slice() {
      [Ok(vi)] => (vi - 1, None, None),
      [Ok(vi), Ok(ti)] => (vi - 1, Some(ti - 1), None),
      [Ok(vi), Ok(ti), Ok(ni)] => (vi - 1, Some(ti - 1), Some(ni - 1)),
      [Ok(vi), Err(_), Ok(ni)] => (vi - 1, None, Some(ni - 1)),
      _ => anyhow::bail!("Invalid face: {:?}", indices),
    };

    res.push(vert)
  }

  Ok(res)
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

fn parse_point3(vs: &[&str]) -> Result<Vec3> {
  Ok(Vec3::from_slice(&parse_floats(vs)?))
}
fn parse_vec3(vs: &[&str]) -> Result<Vec3> {
  Ok(Vec3::from_slice(&parse_floats(vs)?))
}
fn parse_vec2(vs: &[&str]) -> Result<Vec2> {
  Ok(Vec2::from_slice(&parse_floats(vs)?))
}

fn parse_texture(options: &[&str], rel_path: &Path) -> Result<Image<Color>> {
  match options {
    &[path] => load_image_texture(path, rel_path),
    &["-bm", val, path] => {
      let multiplier = parse_float(val)?;
      let mut texture = load_image_texture(path, rel_path)?;
      texture.map_in_place(|color| {
        *color -= 0.5;
        *color *= multiplier;
      });
      Ok(texture)
    }
    _ => bail!("Texture syntax unsupported"),
  }
}

fn load_image_texture(path: &str, rel_path: &Path) -> Result<Image<Color>> {
  let path = rel_path.join(path);
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
  pub meshes: Vec<Rc<Mesh>>,
  pub textures: Rc<TextureStash>,
}

pub fn load(path: &Path) -> Result<MeshObject> {
  let obj = Obj::load(path)?;
  Ok(MeshObject {
    textures: Rc::new(obj.mtl.textures),
    meshes: obj.objs.into_iter().map(|x| Rc::new(x)).collect(),
  })
}
