//! mnist library

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, Read};
use std::vec::Vec;

// MNIST webpage is here: http://yann.lecun.com/exdb/mnist/

pub struct MNist {
    pub data: Vec<f32>,
    pub label: i32,
}

/// Reads mnist data. When appends_bias is true, 1.0 is appended in the last of data.
pub fn read_mnist(image_file_name: &str, label_file_name: &str, appends_bias: bool) -> io::Result<Vec<MNist>> {
    let mut res = Vec::<MNist>::new();

    // read labels
    let mut labels = Vec::<i32>::new();
    {
        let mut f = try!(File::open(label_file_name));

        let magic = try!(f.read_u32::<BigEndian>());
        assert_eq!(magic, 0x0801);

        let num_elems = try!(f.read_u32::<BigEndian>()) as usize;
        assert!(num_elems > 0);

        for b in f.bytes().take(num_elems) {
            let label = try!(b);
            assert!(label < 10);
            labels.push(label as i32);
        }

        assert!(labels.len() == num_elems);
    }

    // read image file
    {
        let mut f = try!(File::open(image_file_name));

        let magic = try!(f.read_u32::<BigEndian>());
        assert_eq!(magic, 0x0803);

        let num_elems = try!(f.read_u32::<BigEndian>()) as usize;
        assert!(num_elems > 0);
        assert_eq!(labels.len(), num_elems);

        let row = try!(f.read_u32::<BigEndian>()) as usize;
        assert!(row == 28);

        let col = try!(f.read_u32::<BigEndian>()) as usize;
        assert!(col == 28);

        for i in 0..num_elems {
            let mut buf: [u8; 28 * 28] = [0; 28 * 28];
            try!(f.read_exact(&mut buf));
            let mut mnist = MNist {
                data: Vec::<f32>::with_capacity(if appends_bias { 28 * 28 + 1 } else { 28 * 28 }),
                label: labels[i],
            };
            for j in 0..28 * 28 {
                mnist.data.push((buf[j] as f32) / 255.0);
            }
            if appends_bias {
                mnist.data.push(1.0);
            }
            res.push(mnist);
        }
    }

    Ok(res)
}
