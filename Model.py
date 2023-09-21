import torch
from torch import nn
from torch.nn import LSTM,AdaptiveAvgPool1d,Linear

class LLM_LSTM(nn.Module):
    def __init__(self, emb_output_dim, num_classes, hidden_size,dr):
        super().__init__()
        self.lstm = LSTM(emb_output_dim, hidden_size, num_layers=2, dropout=dr)
        self.adaptiveAveragePool = AdaptiveAvgPool1d(1)
        self.fc_layer = Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dr)


    def forward(self, embedding, mask):
        mask = torch.unsqueeze(mask, dim=-1)
        llm_embedding_masked = embedding * mask
        out, last_hiddenState = self.lstm(llm_embedding_masked)
        out = out.transpose(2, 1)
        average_poolOut = self.adaptiveAveragePool(out)
        squeezeOut = torch.squeeze(average_poolOut, -1)
        dropout_out1 = self.dropout(squeezeOut)
        fc1_out = self.fc_layer(dropout_out1)
        return fc1_out


class LLM_CNN(nn.Module):
    def __init__(self,filter_widths, num_conv_layers, num_filters, num_classes, dr, max_length, intermediate_pool_size):
        super().__init__()
        self.big_box_encoder = []
        for width in filter_widths:
            self.small_box_encoder = nn.Sequential()
            for layer_indx in range(num_conv_layers):
                self.small_box_encoder.add_module(f"dropout{layer_indx}", nn.Dropout(p=dr))
                inputChannel = max_length if layer_indx == 0 else num_filters
                self.small_box_encoder.add_module(f"Conv1d {layer_indx}", nn.Conv1d(inputChannel, num_filters, width))
                self.small_box_encoder.add_module(f"ReLU {layer_indx}", nn.ReLU())
                self.small_box_encoder.add_module(f"Adaptive_Maxpool {layer_indx}", nn.MaxPool1d(intermediate_pool_size)
                if layer_indx < num_conv_layers - 1 else nn.AdaptiveMaxPool1d(1))
            self.big_box_encoder.append(self.small_box_encoder)
        self.linearLayer = nn.Linear(len(filter_widths)* num_filters, num_classes)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)


    def forward(self, llm_embedding, mask):
        mask = torch.unsqueeze(mask, dim=-1)
        llm_embedding_masked = llm_embedding * mask
        embedded = torch.squeeze(llm_embedding_masked)
        result_list = []
        for indx in range(len(self.big_box_encoder)):
            current_smaller_encoder = self.big_box_encoder[indx]
            output = current_smaller_encoder(embedded)
            result_list.append(output)
        finalConcatenatedTensor = torch.cat(result_list,dim=1)
        SqueezeOut = torch.squeeze(finalConcatenatedTensor, dim=-1)
        linearLayerOut = self.linearLayer(SqueezeOut)
        return linearLayerOut