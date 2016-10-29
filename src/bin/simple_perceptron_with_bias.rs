extern crate rand;
extern crate online_machine_learning;

use online_machine_learning::perceptron::Perceptron;
use online_machine_learning::mnist;
use rand::Rng;

fn train(ith: usize, perceptrons: &mut [Perceptron], train_data: &[mnist::MNist]) {
    let mut correct = 0;
    let mut wrong = 0;
    for td in train_data.iter() {
        for i in 0 .. 10 {
            if perceptrons[i].learn(&td.data, (td.label as usize) == i, 0.001) {
                correct += 1;
            } else {
                wrong += 1;
            }
        }
    }

    println!("TRAIN: count {}: correct={} wrong={}", ith, correct, wrong);
}

fn run_test(perceptrons: &[Perceptron], test_data: &[mnist::MNist]) {
    let mut correct = 0;
    let mut wrong = 0;
    let mut matrix: [[usize; 10]; 10] = [[0; 10]; 10];

    for td in test_data.iter() {
        let mut result_index = 0;
        let mut result_confidence = perceptrons[0].predict(&td.data);
        for i in 1 .. 10 {
            let confidence = perceptrons[i].predict(&td.data);
            if confidence > result_confidence {
                result_index = i as i32;
                result_confidence = confidence;
            }
        }

        matrix[td.label as usize][result_index as usize] += 1;
        if result_index == td.label {
            correct += 1;
        } else {
            wrong += 1;
        }
    }

    println!(" TEST: correct={} wrong={}", correct, wrong);
    for i in 0 .. 10 {
        let mut sum = 0;
        for j in 0 .. 10 {
            sum += matrix[i][j];
            print!("{:>6} ", matrix[i][j]);
        }
        let ratio = (matrix[i][i] as f32) / (sum as f32);
        println!("  {:.2}%", ratio * 100.0);
    }
}

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

    let mut train_data = mnist::read_mnist_image_label_with_bias(&args[1], &args[2]).unwrap();
    println!("train data read ok");

    rand::thread_rng().shuffle(&mut train_data);
    println!("train data is shuffled");

    let test_data = mnist::read_mnist_image_label_with_bias(&args[3], &args[4]).unwrap();
    println!("test data read ok");

    let mut perceptrons = std::vec::Vec::new();
    for _ in 0 .. 10 {
        perceptrons.push(Perceptron::new(28 * 28 + 1));
    }

    for cnt in 0 .. 100 {
        if cnt % 5 == 0 {
            run_test(&perceptrons, &test_data)
        }

        train(cnt, &mut perceptrons, &train_data);
    }

    run_test(&perceptrons, &test_data);
}
