import torch
import torch.nn as nn
import torch.nn.functional as F

class DMSQQuantizer(nn.Module):
    def __init__(self, model, precision_map, device='cpu'):
        """
        Initialize the DMSQ quantizer as a PyTorch module.

        Parameters:
        - model: The pre-trained PyTorch model to be quantized.
        - precision_map: Dictionary mapping layers to desired bit precision.
          Example: {'layer1': 8, 'layer2': 4, 'default': 8}.
        - device: Device to run the quantized model on (e.g., 'cpu', 'cuda').
        """
        super(DMSQQuantizer, self).__init__()
        self.model = model.to(device)
        self.precision_map = precision_map
        self.device = device
        self.scaling_factors = {}
        self.zero_points = {}
        self.original_weights = {}
        self.sensitivity_scores = {}

    def calculate_scaling(self, tensor, bits):
        """
        Calculate scaling factor and zero point for quantization.
        """
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        min_val, max_val = tensor.min(), tensor.max()

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = torch.round(zero_point).to(torch.int)
        
        return scale, zero_point

    def quantize_tensor(self, tensor, scale, zero_point, bits):
        """
        Quantize a tensor to the specified bit precision.
        """
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        return quantized

    def dequantize_tensor(self, quantized_tensor, scale, zero_point):
        """
        Dequantize a tensor from quantized representation.
        """
        return scale * (quantized_tensor - zero_point)

    def layer_sensitivity_analysis(self, inputs, criterion):
        """
        Analyze the sensitivity of each layer to precision loss.
        """
        self.model.eval()
        original_output = self.model(inputs)
        loss_original = criterion(original_output, original_output.clone().detach())

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    precision = self.precision_map.get(name, self.precision_map.get('default', 8))
                    scale, zero_point = self.calculate_scaling(module.weight, precision)
                    quantized_weight = self.quantize_tensor(module.weight, scale, zero_point, precision)
                    dequantized_weight = self.dequantize_tensor(quantized_weight, scale, zero_point)
                    
                    # Replace and evaluate
                    original_weight = module.weight.clone()
                    module.weight = nn.Parameter(dequantized_weight)
                    quantized_output = self.model(inputs)
                    loss_quantized = criterion(original_output, quantized_output)
                    
                    # Restore original weights and record sensitivity
                    module.weight = nn.Parameter(original_weight)
                    self.sensitivity_scores[name] = loss_quantized.item() - loss_original.item()

    def workload_optimization(self, target_memory, target_latency, inputs):
        """
        Optimize precision map based on memory and latency constraints.
        """
        memory_usage = 1e8  # Example: 100MB
        latency_usage = 10  # Example: 10ms
    
        memory_usage = float(memory_usage)
        latency_usage = float(latency_usage)
    
        for name in self.sensitivity_scores:
            if memory_usage > target_memory or latency_usage > target_latency:
                self.precision_map[name] = max(4, self.precision_map.get(name, 8) - 1)
    
            memory_usage *= 0.9
            latency_usage *= 0.95

    def quantize_model(self):
        """
        Apply quantization to the model based on precision_map and sensitivity scores.
        """
        for name, module in self.model.named_modules():
            precision = self.precision_map.get(name, self.precision_map.get('default', 8))
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                with torch.no_grad():
                    self.original_weights[name] = module.weight.clone()
                    
                    scale, zero_point = self.calculate_scaling(module.weight, precision)
                    self.scaling_factors[name] = scale
                    self.zero_points[name] = zero_point

                    quantized_weight = self.quantize_tensor(module.weight, scale, zero_point, precision)
                    module.weight = nn.Parameter(self.dequantize_tensor(quantized_weight, scale, zero_point))

    def quantize_activations(self, activation, name, bits):
        """
        Quantize activations during the forward pass.
        """
        scale, zero_point = self.calculate_scaling(activation, bits)
        quantized_activation = self.quantize_tensor(activation, scale, zero_point, bits)
        return self.dequantize_tensor(quantized_activation, scale, zero_point)

    def forward(self, input_tensor):
        """
        Forward pass through the quantized model with activation quantization.
        """
        def hook(module, input, output):
            if module in self.sensitivity_scores:
                precision = self.precision_map.get(module, self.precision_map.get('default', 8))
                return self.quantize_activations(output, module, precision)
            return output

        handles = []
        for name, module in self.model.named_modules():
            handles.append(module.register_forward_hook(hook))
        
        output = self.model(input_tensor)

        for handle in handles:
            handle.remove()
        
        return output