import hydra
import omlet.utils as U


@hydra.main('./test_hydra.yaml', strict=False)
def main(cfg):
    U.initialize_omlet_config(cfg)
    print('override_name:', cfg.omlet.job.override_name)
    U.print_config(cfg)
    print(cfg.lr, cfg.num_gpus, cfg.batch_size, cfg.run_name)


if __name__ == '__main__':
    main()