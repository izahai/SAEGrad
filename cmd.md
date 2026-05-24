python3 -m pytest mesd/test/test_layer_select.py -q

# Train:
CUDA_VISIBLE_DEVICES=0 python main.py --erase_concept 'Shiba Inu' --train_method 'esd-x' --save_gradient mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight