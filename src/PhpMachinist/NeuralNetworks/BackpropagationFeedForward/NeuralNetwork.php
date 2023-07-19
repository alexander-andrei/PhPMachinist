<?php

namespace PhpMachinist\NeuralNetworks\BackpropagationFeedForward;

class NeuralNetwork
{
    /**
     * @var Neuron[]
     */
    private array $outputLayer;

    /**
     * @var Neuron[]
     */
    private array $hiddenLayer;
    private int $inputLayerSize;
    private int $hiddenLayerSize;
    private int $outputLayerSize;
    private float $learningRate;

    public function __construct(int $inputLayerSize, int $hiddenLayerSize, int $outputLayerSize, float $learningRate)
    {
        $this->inputLayerSize = $inputLayerSize;
        $this->hiddenLayerSize = $hiddenLayerSize;
        $this->outputLayerSize = $outputLayerSize;

        // Create hidden layer neurons
        $this->hiddenLayer = [];
        for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
            $this->hiddenLayer[] = new Neuron($this->inputLayerSize);
        }

        // Create output layer neurons
        $this->outputLayer = [];
        for ($i = 0; $i < $this->outputLayerSize; $i++) {
            $this->outputLayer[] = new Neuron($this->hiddenLayerSize);
        }

        $this->learningRate = $learningRate;
    }

    // can use sigmoid instead of derivative
    private function sigmoid(float $x): float
    {
        return 1 / (1 + exp(-$x));
    }

    private function sigmoidDerivative(float $x): float
    {
        return $x * (1 - $x);
    }

    public function forwardPropagation(array $input): array
    {
        return $this->calculateOutput($this->calculateHiddenOutput($input));
    }

    /**
     * Backpropagation is calculated as follows:
     *
     * 1. During forward propagation, the input data passes through the neural network's
     * layers, and the output is computed. Each neuron's output is determined by applying
     * the activation function (usually the sigmoid function) to the weighted sum of its inputs.
     *
     * 2. The loss function measures the error between the predicted outputs from the neural
     * network and the actual targets. In this example, we'll use the mean squared error (MSE)
     * loss function, which is often used for regression tasks.
     *
     * 3. The key idea of backpropagation is to compute the gradients of the loss function with
     * respect to the weights and biases of the neural network. These gradients indicate how
     * much the loss function will change concerning small changes in the weights and biases.
     * We use these gradients to update the model's parameters to minimize the loss.
     *
     * Formula: https://www.wikiwand.com/en/Backpropagation (all formulas for above steps can be found on the wiki)
     */
    public function backPropagation($input, $target): void
    {
        $hiddenOutput = $this->calculateHiddenOutput($input);
        $output = $this->calculateOutput($hiddenOutput);


        $outputDeltas = [];
        for ($i = 0; $i < $this->outputLayerSize; $i++) {
            $error = $target[$i] - $output[$i];
            $outputDeltas[] = $error * $this->sigmoidDerivative($output[$i]);
        }

        $hiddenDeltas = [];
        for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
            $error = 0;
            for ($j = 0; $j < $this->outputLayerSize; $j++) {
                $error += $outputDeltas[$j] * $this->outputLayer[$j]->getWeights()[$i];
            }
            $hiddenDeltas[] = $error * $this->sigmoidDerivative($hiddenOutput[$i]);
        }

        for ($i = 0; $i < $this->outputLayerSize; $i++) {
            $weights = $this->outputLayer[$i]->getWeights();
            for ($j = 0; $j < $this->hiddenLayerSize; $j++) {
                $weights[$j] += $this->learningRate * $outputDeltas[$i] * $hiddenOutput[$j];
            }
            $this->outputLayer[$i]->setWeights($weights);

            $bias = $this->outputLayer[$i]->getBias();
            $bias += $this->learningRate * $outputDeltas[$i];
            $this->outputLayer[$i]->setBias($bias);
        }

        for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
            $weights = $this->hiddenLayer[$i]->getWeights();
            for ($j = 0; $j < $this->inputLayerSize; $j++) {
                $weights[$j] += $this->learningRate * $hiddenDeltas[$i] * $input[$j];
            }
            $this->hiddenLayer[$i]->setWeights($weights);

            $bias = $this->hiddenLayer[$i]->getBias();
            $bias += $this->learningRate * $hiddenDeltas[$i];


            $this->hiddenLayer[$i]->setBias($bias);
        }
    }

    private function calculateHiddenOutput(array $input): array
    {
        $hiddenOutput = [];
        for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
            $hiddenOutput[] = $this->hiddenLayer[$i]->compute($input);
        }

        return $hiddenOutput;
    }

    private function calculateOutput(array $hiddenOutput): array
    {
        $output = [];
        for ($i = 0; $i < $this->outputLayerSize; $i++) {
            $output[] = $this->outputLayer[$i]->compute($hiddenOutput);
        }

        return $output;
    }
}
