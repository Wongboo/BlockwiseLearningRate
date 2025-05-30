from transformers import LlamaConfig, LlamaForCausalLM

def build_llama_models(parameter_number, d_input, max_seq_length, device):
    extra_config = {}
    # python 3.8 does not have match
    if True:
        if parameter_number == "90M":
            d_model = 768  ## fixed due to d_{kv}
            num_heads = 12  ## fixed due to d_{kv}
            num_layers = 2
            d_ff = d_model * 4
            dropout = 0.0

        if parameter_number == "134M":
            d_model = 768  ## fixed due to d_{kv}
            num_heads = 12  ## fixed due to d_{kv}
            num_layers = 6
            d_ff = d_model * 4
            dropout = 0.0

        if parameter_number == "0.23B":
            d_model = 768 ## fixed due to d_{kv}
            num_heads = 12  ## fixed due to d_{kv}
            num_layers = 16
            d_ff = d_model * 4
            dropout = 0.0

        if parameter_number == "0.25B":
            d_model = 1024  ## fixed due to d_{kv}
            num_heads = 16  ## fixed due to d_{kv}
            num_layers = 8
            d_ff = d_model * 4
            dropout = 0.0

        if parameter_number == "0.5B":
            d_model = 1280  ## fixed due to d_{kv}
            num_heads = 20  ## fixed due to d_{kv}
            num_layers = 15
            d_ff = d_model * 4
            dropout = 0.0

        if parameter_number == "0.75B":
            d_model = 1664  ## fixed due to d_{kv}
            num_heads = 26  ## fixed due to d_{kv}
            num_layers = 13
            d_ff = d_model * 4
            dropout = 0.0

        if parameter_number == "0.9B": 
            d_model = 1600  ## fixed due to d_{kv}
            num_heads = 25  ## fixed due to d_{kv}
            num_layers = 18
            d_ff = d_model * 4
            dropout = 0.0
        
        if parameter_number == "TinyLlama":
            d_model = 2048
            num_heads = 32
            num_layers = 22
            d_ff = 5632
            dropout = 0.0
            max_seq_length = 2048
            extra_config = {'num_key_value_heads': 4, 'rms_norm_eps': 1e-5}
            # model_args = LlamaConfig(vocab_size=d_input, hidden_size=2048, intermediate_size=5632, num_attention_heads=32, num_hidden_layers=22, num_key_value_heads=4, rms_norm_eps=1e-5)
            # return LlamaForCausalLM(model_args).to(device)
        
        if parameter_number == "TinyLlama2":
            d_model = 2048
            num_heads = 32
            num_layers = 22
            d_ff = 5632
            dropout = 0.0
            extra_config = {'num_key_value_heads': 4, 'rms_norm_eps': 1e-5}
            # model_args = LlamaConfig(vocab_size=d_input, hidden_size=2048, intermediate_size=5632, num_attention_heads=32, num_hidden_layers=22, num_key_value_heads=4, rms_norm_eps=1e-5)
            # return LlamaForCausalLM(model_args).to(device)
            
    model_args = LlamaConfig(vocab_size=d_input, hidden_size=d_model, num_attention_heads=num_heads, attention_dropout=dropout, num_hidden_layers=num_layers, intermediate_size=d_ff, max_position_embeddings=max_seq_length, **extra_config)
    # print(model_args)
    # model_args._attn_implementation = 'sdpa'
    # print(model_args._attn_implementation)
    return LlamaForCausalLM(model_args).to(device)