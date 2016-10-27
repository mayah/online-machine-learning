//! mnist library

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, Read};
use std::vec::Vec;

// MNIST webpage is here: http://yann.lecun.com/exdb/mnist/

pub struct MNist {
    pub data: [f32; 28 * 28],
    pub label: i32,
}

pub fn read_mnist_image_label(image_file_name: &str, label_file_name: &str) -> io::Result<Vec<MNist>> {
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

        for i in 0 .. num_elems {
            let mut buf: [u8; 28 * 28] = [0; 28 * 28];
            try!(f.read_exact(&mut buf));
            let mut data: [f32; 28 * 28] = [0.0; 28 * 28];
            for j in 0 .. 28 * 28 {
                data[j] = (buf[j] as f32) / 255.0
            }
            res.push(MNist {
                data: data,
                label: labels[i],
            });
        }
    }

    Ok(res)
}
