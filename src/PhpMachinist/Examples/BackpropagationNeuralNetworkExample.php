<?php

require 'vendor/autoload.php';

use PhpMachinist\NeuralNetworks\BackpropagationFeedForward\NeuralNetwork;

$inputLayerSize = 2;
$hiddenLayerSize = 3;
$outputLayerSize = 1;

$neuralNetwork = new NeuralNetwork($inputLayerSize, $hiddenLayerSize, $outputLayerSize, 0.1);

$trainingInputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
$trainingTargets = [[0], [1], [1], [0]];

$epochs = 10000;
for ($i = 0; $i < $epochs; $i++) {
    for ($j = 0; $j < count($trainingInputs); $j++) {
        $input = $trainingInputs[$j];
        $target = $trainingTargets[$j];
        $neuralNetwork->backPropagation($input, $target);
    }
}


foreach ($trainingInputs as $input) {
    $output = $neuralNetwork->forwardPropagation($input);
    echo "Input: " . implode(", ", $input) . " | Output: " . round($output[0]) . PHP_EOL;
}