<?php

namespace PhpMachinist\NeuralNetworks\FeedForward;

/**
 * FeedForward is the most basic type of neural network, where the information flows in
 * one direction, from the input layer to the output layer. There are no
 * feedback connections, and the network is used for tasks like
 * classification and regression.
 */
class NeuralNetwork {
    private array $layers;

    /**
     * @param $layerSizes => an array representing the # of neurons in each layer
     *                    => [2, 3, 1]
     */
    public function __construct($layerSizes) {
        $this->layers = [];

        for ($x = 0; $x < count($layerSizes); $x++) {
            /**
             * In a feedforward neural network, each neuron in a hidden layer receives
             * inputs from all neurons in the previous layer, including a bias input.
             * The bias input is an additional constant input typically set to 1, which
             * allows the neuron to learn an offset or intercept term.
             *
             * To account for this bias input, we add 1 to the number of inputs in the
             * hidden layers (i.e., $layerSizes[$x - 1] + 1). For the first hidden layer
             * $x equals 1, so $layerSizes[$x - 1] represents the number of neurons in
             * the input layer. Adding 1 to this value accounts for the bias input.
             *
             * The equation ensures that each neuron in the hidden layers has the correct
             * number of weights, including the bias weight. The bias weight will be
             * associated with the additional bias input, allowing the neuron to learn
             * an offset term during training.
             *
             * $numInputs = [2, 3, 4]
             */
            $numInputs = ($x === 0) ? $layerSizes[$x] : $layerSizes[$x - 1] + 1;
            $numNeurons = $layerSizes[$x];

            $layer = [];

            /**
             * Create neurons for each layer:
             *
             * [
             *      2, => 1st layer with 2 neurons
             *      3, => 2nd layer with 3 neurons
             *      1, => 3rd layer with 1 neuron
             * ]
             *
             */
            for ($y = 0; $y < $numNeurons; $y++) {
                $layer[] = new Neuron($numInputs);
            }

            $this->layers[] = $layer;
        }
    }

    /**
     * @param $inputs => array of [age, height] => [25, 170]
     *
     */
    public function forward($inputs) {
        $outputs = $inputs;

        /**
         * Activate neuron in each layer (as mentioned in comments above, we have 3 layers)
         */
        foreach ($this->layers as $layer) {
            $newOutputs = [];

            /** @var Neuron $neuron */
            foreach ($layer as $neuron) {
                $newOutputs[] = $neuron->activate($outputs);
            }
            $outputs = $newOutputs;
        }

        return $outputs;
    }
}
