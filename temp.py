def train_model(func_name,start_rand_idxs=None,curr_epoch=num_epochs, bud=None):

    idxs = start_rand_idxs

    criterion = nn.MSELoss()
    
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    main_model = RegressionNet(M)
    main_model.apply(weight_reset)
   
    main_model = main_model.to(device)
    #criterion_sum = nn.MSELoss(reduction='sum')
    #main_optimizer = optim.SGD(main_model.parameters(), lr=learning_rate)
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change, gamma=0.5)
    scheduler = optim.lr_scheduler.StepLR(main_optimizer, step_size=1, gamma=0.1)

    if func_name == 'Random':
        print("Starting Random Run!")
    elif func_name == 'Random with Prior':
        print("Starting Random with Prior Run!")

    idxs.sort()
    np.random.seed(42)
    np_sub_idxs = np.array(idxs)
    np.random.shuffle(np_sub_idxs)
    loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
            transform=None),shuffle=False,batch_size=train_batch_size)

    stop_count = 0
    prev_loss = 1000
    prev_loss2 = 1000
    i =0
    mul=1
    lr_count = 0
    while(True):
        #for i in range(curr_epoch):#num_epochs):
        
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[idxs], y_trn[idxs]

        temp_loss = 0.

        for batch_idx in list(loader_tr.batch_sampler):
            
            inputs, targets = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)
            main_optimizer.zero_grad()
            
            
            scores = main_model(inputs)

            #l2_reg = 0
            #for param in main_model.parameters():
            #    l2_reg += torch.norm(param)

            l = [torch.flatten(p) for p in main_model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)
            
            loss = criterion(scores, targets) +  reg_lambda*l2_reg*len(batch_idx)
            temp_loss += loss.item()
            loss.backward()

            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            print(prev_loss,temp_loss,mul)
            #print(main_optimizer.param_groups[0]['lr'])

        if abs(prev_loss - temp_loss) <= 1e-1*mul or prev_loss2 == temp_loss:
            #print(main_optimizer.param_groups[0]['lr'])
            lr_count += 1
            if lr_count == 10:
                #print(i,"Reduced")
                print(prev_loss,temp_loss,main_optimizer.param_groups[0]['lr'])
                scheduler.step()
                mul/=10
                lr_count = 0
        else:
            lr_count = 0

        if abs(prev_loss - temp_loss) <= 1e-3 and stop_count >= 10:
            print(i)
            break 
        elif abs(prev_loss - temp_loss) <= 1e-3:
            stop_count += 1
        else:
            stop_count = 0

        if i>=2000:
            break

        #print(temp_loss,prev_loss)
        prev_loss2 = prev_loss
        prev_loss = temp_loss
        i+=1

    #tst_accuracy = torch.zeros(len(x_tst_list))
    #val_accuracy = torch.zeros(len(x_val_list))

    loader_val = DataLoader(CustomDataset(x_val, y_val,transform=None),shuffle=False,\
        batch_size=batch_size)

    loader_tst = DataLoader(CustomDataset(x_tst, y_tst,transform=None),shuffle=False,\
        batch_size=batch_size)
    
    main_model.eval()
    with torch.no_grad():
        '''full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("\nFinal SubsetTrn and FullTrn Loss:", sub_trn_loss.item(),full_trn_loss.item(),file=logfile)'''
        
        val_loss = 0.
        for batch_idx in list(loader_val.batch_sampler):
            
            inputs, targets = loader_val.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)
        
            val_out = main_model(inputs)
            val_loss += criterion(val_out, targets)
        
        val_loss /= len(loader_val.batch_sampler)

        test_loss = 0.
        for batch_idx in list(loader_tst.batch_sampler):
            
            inputs, targets = loader_tst.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = main_model(inputs)
            test_loss += criterion(outputs, targets)

        test_loss /= len(loader_tst.batch_sampler)        

    return [val_loss,test_loss] #[val_accuracy, val_classes, tst_accuracy, tst_classes]
