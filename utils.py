def make_experiment(crayon, name, fold, reopen=False):
    if reopen:
        train_exp = crayon.open_experiment(f'{name}_f{fold}_train')
        valid_exp = crayon.open_experiment(f'{name}_f{fold}_valid')
    else:
        try:
            crayon.remove_experiment(f'{name}_f{fold}_train')
        except ValueError:
            pass
        try:
            crayon.remove_experiment(f'{name}_f{fold}_valid')
        except ValueError:
            pass
        train_exp = crayon.create_experiment(f'{name}_f{fold}_train')
        train_exp.scalar_steps['lr'] = 1
        valid_exp = crayon.create_experiment(f'{name}_f{fold}_valid')
    return train_exp, valid_exp
