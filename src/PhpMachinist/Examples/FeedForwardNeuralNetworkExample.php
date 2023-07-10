<?php

require 'vendor/autoload.php';

use PhpMachinist\NeuralNetworks\FeedForward\NeuralNetwork;

$layerSizes = [2, 3, 1]; // Number of neurons in each layer
$network = new NeuralNetwork($layerSizes);

$inputs = [25, 170]; // Input values

$outputs = $network->forward($inputs);

echo "Outputs: ";
foreach ($outputs as $output) {
    echo "$output ";
}
