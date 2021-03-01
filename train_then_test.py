import argparse
from utils_ic import load_data, read_jason
from model_ic import NN_Classifier, validation, make_NN, save_checkpoint, test_model

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")
parser.add_argument("--print_model", default=False, help="print model")
parser.add_argument("--use_pretrain", default=True, help="use pretrained model")
parser.add_argument("--train_whole", default=False, help="train the whole model")
parser.add_argument("--print_every", type=int, default=40, help="step interval to print")
parser.add_argument("--train_custom", default=False, help="train a custom model")
parser.add_argument("--num_layers", type=int, default=1, help="number of layers")

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

if args.use_pretrain == 'False':
    args.use_pretrain = False
elif args.use_pretrain == 'True':
    args.use_pretrain = True
if args.train_whole == 'False':
    args.train_whole = False
elif args.train_whole == 'True':
    args.train_whole = True
if args.train_custom == 'False':
    args.train_custom = False
elif args.train_custom == 'True':
    args.train_custom = True

model = make_NN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data, print_model=args.print_model, \
                use_pretrain=args.use_pretrain, train_whole=args.train_whole, print_every=args.print_every,
                train_custom=args.train_custom, num_layers=args.num_layers)

test_model(model, testloader)

if args.save_dir:
    save_checkpoint(model, args.save_dir)