python -m src.train \
    --config configs/mka_tcresnet.yaml \
    --limit 2000 \
    --epochs 5 \
    --output outputs/tcresnet_smoke_v2


python -m src.train \
    --config configs/mka_tcresnet.yaml \
    --epochs 20 \
    --output outputs/tcresnet_e20_full_v2