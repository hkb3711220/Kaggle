import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import *
from preprocess import *
from model import InchiModel
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_file_path(image_id):
    return "../input/bms-molecular-translation/test/" + "{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )


def inference(test_loader, model, tokenizer, device, tta=True):
    model.eval()
    all_text_predictions = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        with torch.no_grad():
            images = images.to(device).float()
            if tta:
                p = []
                logit = model.infer(images)
                p.append(F.softmax(logit, -1))

                logit = model.infer(torch.flip(images, dims=(2,)).contiguous())
                p.append(F.softmax(logit, -1))

                logit = model.infer(torch.flip(images, dims=(3,)).contiguous())
                p.append(F.softmax(logit, -1))

                logit = model.infer(torch.flip(images, dims=(2, 3)).contiguous())
                p.append(F.softmax(logit, -1))

                logit = model.infer(images.permute(0, 1, 3, 2).contiguous())
                p.append(F.softmax(logit, -1))

                p = torch.stack(p).mean(0)
                predictions = torch.argmax(p, -1)
            else:
                predictions = model.predict(images)
        predictions = predictions.detach().cpu().numpy()
        text_predictions = tokenizer.predict_captions(predictions)
        all_text_predictions.append(text_predictions)
    text_preds = np.concatenate(all_text_predictions)

    return text_preds


if __name__ == '__main__':
    import config

    tokenizer = torch.load('tokenizer.pth')
    print(f"tokenizer.stoi: {tokenizer.stoi}")
    model = InchiModel(vocab_size=len(tokenizer.stoi))
    model.to(device)
    weight_file ="./effnet_transformers_v3/best-checkpoint-1449999.bin"
    model_state = torch.load(weight_file, map_location=device)
    model.load_state_dict(model_state['model'])
    test = pd.read_csv('../input/bms-molecular-translation/sample_submission.csv')[400000:]
    test['file_path'] = test['image_id'].apply(get_test_file_path)
    print(f'test.shape: {test.shape}')
    test_dataset = TestDataset(test, transform=get_transforms(config, mode="valid"))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=6)
    predictions = inference(test_loader, model, tokenizer, device)
    test['InChI'] = [f"InChI=1S/{text}" for text in predictions]
    test[['image_id', 'InChI']].to_csv('submission_400000-.csv', index=False)
