<?php

require 'vendor/autoload.php';

// Training data
use PhpMachinist\Models\SVM;

$samples = [
    [18, 160],
    [25, 165],
    [32, 170],
    [45, 175],
    [60, 180]
];
$labels = [0, 0, 1, 1, 1]; // 0 represents short, 1 represents tall

// Create and train the SVM classifier
$classifier = new SVM('polynomial');
$classifier->train($samples, $labels);

// Predict the class for a new sample
$newSample = [32, 0]; // 0 represents an unknown label
$predictedLabel = $classifier->predict($newSample);

echo "Predicted label: " . ($predictedLabel === 0 ? 'short' : 'tall');