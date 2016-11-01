extern crate rand;
extern crate online_machine_learning;
extern crate getopts;

use getopts::Options;
use online_machine_learning::arow::Arow;
use online_machine_learning::logistic::LogisticRegression;
use online_machine_learning::linear_classifier::LinearClassifier;
use online_machine_learning::perceptron::Perceptron;
use online_machine_learning::svm::SVM;
use online_machine_learning::mnist;
use rand::Rng;
use std::boxed::Box;

fn train(ith: usize, perceptrons: &mut [Box<LinearClassifier>], train_data: &[mnist::MNist]) {
    let mut correct = 0;
    let mut wrong = 0;
    for td in train_data.iter() {
        for i in 0..10 {
            if perceptrons[i].learn(&td.data, (td.label as usize) == i, 0.001) {
                correct += 1;
            } else {
                wrong += 1;
            }
        }
    }

    println!("TRAIN: count {}: correct={} wrong={}", ith, correct, wrong);
}

fn run_test(perceptrons: &[Box<LinearClassifier>], test_data: &[mnist::MNist]) {
    let mut correct = 0;
    let mut wrong = 0;
    let mut matrix: [[usize; 10]; 10] = [[0; 10]; 10];

    for td in test_data.iter() {
        let mut result_index = 0;
        let mut result_confidence = perceptrons[0].margin(&td.data);
        for i in 1..10 {
            let confidence = perceptrons[i].margin(&td.data);
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
    for i in 0..10 {
        let mut sum = 0;
        for j in 0..10 {
            sum += matrix[i][j];
            print!("{:>6} ", matrix[i][j]);
        }
        let ratio = (matrix[i][i] as f32) / (sum as f32);
        println!("  {:.2}%", ratio * 100.0);
    }
}

fn normalize_train_test(train_data: &mut [mnist::MNist], test_data: &mut [mnist::MNist]) {
    let mut ave = [0.0; 28 * 28];
    for td in train_data.iter() {
        for i in 0..28*28 {
            ave[i] += td.data[i];
        }
    }
    for i in 0..28*28 {
        ave[i] /= train_data.len() as f32;
    }
    for td in train_data.iter_mut() {
        for i in 0..28*28 {
            td.data[i] -= ave[i];
        }
    }
    for td in test_data.iter_mut() {
        for i in 0..28*28 {
            td.data[i] -= ave[i];
        }
    }
}

pub fn main() {
    let args: Vec<_> = std::env::args().collect();

    let mut opts = Options::new();
    opts.optflag("", "bias", "use bias");
    opts.optflag("", "normalization", "use normalization");
    opts.optopt("t", "type", "classifier type. perceptron, svm, logistic, or arow. default is perceptron", "TYPE");

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!(f.to_string()) }
    };

    let uses_bias = matches.opt_present("bias");
    let uses_normalization = matches.opt_present("normalization");
    let classifier_type = matches.opt_str("type").unwrap_or("perceptron".to_owned());

    if matches.free.len() < 4 {
        println!("{} (options...) <train_image_file> <train_label_file> <test_image_file> <test_label_file>", args[0]);
        std::process::exit(1);
    }

    let train_image_file = matches.free[0].clone();
    let train_label_file = matches.free[1].clone();
    let test_image_file = matches.free[2].clone();
    let test_label_file = matches.free[3].clone();

    println!("train image_file: {}", train_image_file);
    println!("train label_file: {}", train_label_file);
    println!(" test image_file: {}", test_image_file);
    println!(" test label_file: {}", test_label_file);

    let mut train_data = mnist::read_mnist(&train_image_file, &train_label_file, uses_bias).unwrap();
    println!("train data read ok");

    rand::thread_rng().shuffle(&mut train_data);
    println!("train data is shuffled");

    let mut test_data = mnist::read_mnist(&test_image_file, &test_label_file, uses_bias).unwrap();
    println!("test data read ok");

    if uses_normalization {
        normalize_train_test(&mut train_data, &mut test_data);
    }

    let mut perceptrons = std::vec::Vec::<Box<LinearClassifier>>::new();
    for _ in 0..10 {
        let n = (28 * 28) + (if uses_bias { 1 } else { 0 });
        let classifier: Box<LinearClassifier> = match classifier_type.as_ref() {
            "perceptron" => Box::new(Perceptron::new(n)),
            "arow" => Box::new(Arow::new(n)),
            "svm" => Box::new(SVM::new(n)),
            "logistic" => Box::new(LogisticRegression::new(n)),
            _ => {
                assert!(false);
                unreachable!()
            }
        };

        perceptrons.push(classifier);
    }

    for cnt in 0..100 {
        if cnt % 5 == 0 {
            run_test(&perceptrons, &test_data)
        }

        train(cnt, &mut perceptrons, &train_data);
    }

    run_test(&perceptrons, &test_data);
}
