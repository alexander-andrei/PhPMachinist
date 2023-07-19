<?php

namespace PhpMachinist\NeuralNetworks\BackpropagationFeedForward;


class Neuron
{
    private array $weights;
    private float $bias;

    public function __construct(int $inputSize)
    {
        $this->weights = [];
        for ($i = 0; $i < $inputSize; $i++) {
            $this->weights[] = rand(-100, 100) / 100.0; // Random weights between -1 and 1
        }
        $this->bias = rand(-100, 100) / 100.0; // Random bias between -1 and 1
    }

    public function compute(array $inputs): float
    {
        $weightedSum = 0;
        for ($i = 0; $i < count($inputs); $i++) {
            $weightedSum += $inputs[$i] * $this->weights[$i];
        }
        $weightedSum += $this->bias;

        return $this->sigmoid($weightedSum);
    }

    private function sigmoid($x): float
    {
        return 1 / (1 + exp(-$x));
    }

    public function getWeights(): array
    {
        return $this->weights;
    }

    public function setWeights($weights): void
    {
        $this->weights = $weights;
    }

    public function getBias(): float
    {
        return $this->bias;
    }

    public function setBias($bias): void
    {
        $this->bias = $bias;
    }
}