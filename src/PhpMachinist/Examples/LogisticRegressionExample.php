<?php

use PhpMachinist\Models\LogisticRegression;

/**
 * First value is age
 * Second value is height
 */
$peopleByAgeAndHeight = [
    [18, 165],
    [25, 170],
    [32, 175],
    [45, 180],
    [60, 185],
    [20, 160],
    [28, 168],
    [35, 172],
    [50, 178],
    [65, 190]
];

/**
 * 0 -> female
 * 1 -> male
 */
$peopleBySex = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1];

$lr = new LogisticRegression();
$lr->train($peopleByAgeAndHeight, $peopleBySex);

$weights = $lr->getWeights();
$intercept = $lr->getIntercept();

echo "Learned Weights: " . implode(", ", $weights) . "\n";
echo "Intercept: $intercept\n";

$newSample = [35, 170];
$prediction = $lr->predict([$newSample])[0];

echo "New Sample: [" . implode(", ", $newSample) . "]\n";
echo "Prediction: " . ($prediction == 0 ? "Female" : "Male") . "\n";