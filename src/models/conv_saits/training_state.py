"""Optional epoch checkpointing for long-running Conv-SAITS studies."""

from dataclasses import asdict

import torch


def resume_training(backend, optimizer, stopping):
    path = backend.training_checkpoint
    if path is None or not path.is_file():
        return 0, False
    saved = torch.load(path, map_location=backend.device, weights_only=False)
    if saved['config'] != asdict(backend.config):
        raise ValueError('Cannot resume Conv-SAITS with a changed configuration.')
    backend.network.load_state_dict(saved['state_dict'])
    optimizer.load_state_dict(saved['optimizer'])
    for key in ('best_state', 'best_score', 'patience_score', 'stale_epochs', 'best_epoch'):
        setattr(stopping, key, saved[key])
    backend.training_history, backend.best_epoch = saved['history'], saved['best_epoch']
    torch.set_rng_state(saved['torch_rng'].cpu())
    if backend.device.type == 'cuda':
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    return saved['epoch'], saved['stopped']


def save_training(backend, optimizer, stopping, epoch, stopped):
    path = backend.training_checkpoint
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        state = {key: getattr(stopping, key) for key in
                 ('best_state', 'best_score', 'patience_score', 'stale_epochs', 'best_epoch')}
        state.update(config=asdict(backend.config), epoch=epoch,
                     state_dict=backend.network.state_dict(), optimizer=optimizer.state_dict(),
                     history=backend.training_history, stopped=stopped,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if backend.device.type == 'cuda' else [])
        torch.save(state, temporary)
        temporary.replace(path)
    if backend.epoch_callback is not None:
        backend.epoch_callback(backend.training_history[-1])
