import io
import pickle


def pass_through_pickle(surf):
    with io.BytesIO() as f:
        pickle.dump(surf, f)
        f.seek(0)
        return pickle.load(f)
