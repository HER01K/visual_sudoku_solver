use std::{fs, io};
use crate::{NUM_INPUTS, NUM_OUTPUTS};

pub type Image = [u8; NUM_INPUTS];

const IMAGES_MAGIC: u32 = 0x0000_0803; // 2051
const LABELS_MAGIC: u32 = 0x0000_0801; // 2049

pub struct Dataset {
    pub images: Vec<Image>,
    pub labels: Vec<u8>,
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

pub fn read_images(path: &str) -> io::Result<Vec<Image>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("images file too short: {} bytes", bytes.len())
        ));
    }

    let magic = read_u32(&bytes, 0);
    if magic != IMAGES_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad images magic {magic:#010x}")
        ));
    }

    let n = read_u32(&bytes, 4) as usize;
    let rows = read_u32(&bytes, 8) as usize;
    let cols = read_u32(&bytes, 12) as usize;
    if rows * cols != NUM_INPUTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected {NUM_INPUTS} pixels per image, got {rows}x{cols}")
        ));
    }

    let data = &bytes[16..];
    if data.len() != n * NUM_INPUTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("header says {n} images but data is {} bytes", data.len())
        ));
    }

    Ok(data
        .chunks_exact(NUM_INPUTS)
        .map(|chunk| chunk.try_into().unwrap())
        .collect())
}

pub fn read_labels(path: &str) -> io::Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("labels file too short: {} bytes", bytes.len())
        ));
    }

    let magic = read_u32(&bytes, 0);
    if magic != LABELS_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad labels magic {magic:#010x}")
        ));
    }

    let n = read_u32(&bytes, 4) as usize;
    let data = &bytes[8..];
    if data.len() != n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("header says {n} labels but data is {} bytes", data.len())
        ));
    }

    if let Some(&bad) = data.iter().find(|&&l| l as usize >= NUM_OUTPUTS) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("label {bad} out of range 0..{NUM_OUTPUTS}")
        ));
    }

    Ok(data.to_vec())
}

pub fn load_dataset(prefix: &str) -> io::Result<Dataset> {
    let images = read_images(format!("{prefix}-images.idx3-ubyte").as_str())?;
    let labels = read_labels(format!("{prefix}-labels.idx1-ubyte").as_str())?;

    if images.len() != labels.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} images but {} labels", images.len(), labels.len())
        ));
    }

    Ok(Dataset { images, labels })
}
