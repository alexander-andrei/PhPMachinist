<?php

use PhpMachinist\Models\LinearRegression;

$regression = new LinearRegression();
$ages = [18, 25, 32, 45, 60];
$heights = [160, 165, 170, 175, 180];
$regression->train($ages, $heights);

$testingData = [35, 10, 14];
$predictions = $regression->predict($testingData);

foreach ($predictions as $index => $prediction) {
    echo "Prediction for age = {$testingData[$index]}: $prediction\n";
}