    '''if(sys.argv[1].strip()=='ask'):
        model = torch.load('saved_transformer.pt')
        print('Model loaded successfully', model)
        while(1):
                question = input("Insert a Question for the model: ")
                tkq=torch.tensor([voc.stoi['<SOS>']])
                tkq= torch.cat((tkq, torch.tensor(voc.numericalize(list(question))), torch.tensor([voc.stoi['<EOS>']])))
                print('passing to the model: ', tkq)
                print('Stringa prodotta: ', tensor_to_string(voc, tkq))
                preds = model()
                print(tensor_to_string(preds[1:-1]))'''
