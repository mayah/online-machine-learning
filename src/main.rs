extern crate byteorder;

pub mod mnist;

pub fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        println!("{} <train_image_file> <train_label_file> <test_image_file> <test_label_file>", args[0]);
        std::process::exit(1);
    }

    println!("train image_file: {}", &args[1]);
    println!("train label_file: {}", &args[2]);
    println!(" test image_file: {}", &args[3]);
    println!(" test label_file: {}", &args[4]);

    let train_data = mnist::read_mnist_image_label(&args[1], &args[2]);
    println!("train data read ok");

    let test_data = mnist::read_mnist_image_label(&args[3], &args[4]);
    println!("test data read ok");
}
