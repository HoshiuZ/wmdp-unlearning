"""Read-only default entry: resolve the retained pure RMU candidate; never train."""
from omegaconf import OmegaConf
from run_rmu_adaptation import configuration


def main():
    cfg = configuration(12, 2, 5e-5, 32, 'pure_rmu_h12_a2_s32_40', 40)
    assert cfg.trainer.handler == 'RMU'
    assert cfg.trainer.method_args.retain_loss_type == 'EMBED_DIFF'
    print(OmegaConf.to_yaml(cfg.trainer))


if __name__ == '__main__':
    main()
