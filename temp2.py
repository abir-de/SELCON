def train_model_fair(func_name,start_rand_idxs=None, bud=None):

    sub_idxs = start_rand_idxs

    criterion = nn.MSELoss()
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    main_model = RegressionNet(M)
    main_model.apply(weight_reset)

    main_model = main_model.to(device)

    #criterion_sum = nn.MSELoss(reduction='sum')
    
    alphas = torch.randn_like(deltas,device=device) + 5 #,requires_grad=True)
    alphas.requires_grad = True
    #print(alphas)
    #alphas = torch.ones_like(deltas,requires_grad=True)
    '''main_optimizer = optim.SGD([{'params': main_model.parameters()},
                {'params': alphas}], lr=learning_rate) #'''
    main_optimizer = torch.optim.Adam([
                {'params': main_model.parameters()}], lr=learning_rate)
                
    dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=learning_rate) #{'params': alphas} #'''

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change,\
    #     gamma=0.5) #[e*2 for e in change]
    scheduler = optim.lr_scheduler.StepLR(main_optimizer, step_size=1, gamma=0.1)

    #alphas.requires_grad = False

    #delta_extend = torch.repeat_interleave(deltas,val_size, dim=0)

    if func_name == 'Random':
        print("Starting Random with fairness Run!")
    elif func_name == 'Fair_subset':

        cached_state_dict = copy.deepcopy(main_model.state_dict())
        alpha_orig = copy.deepcopy(alphas)

        fsubset_d = FindSubset_Vect(x_trn, y_trn, x_val, y_val,main_model,criterion,\
            device,deltas,learning_rate,reg_lambda,batch_size)

        fsubset_d.precompute(int(num_epochs/4),sub_epoch,alpha_orig)

        '''main_model.load_state_dict(cached_state_dict)
        alpha_orig = copy.deepcopy(alphas)

        fsubset = FindSubset(x_trn, y_trn, x_val, y_val,main_model,criterion,\
            device,deltas,learning_rate,reg_lambda)

        fsubset.precompute(int(num_epochs/4),sub_epoch,alpha_orig,batch_size)'''
        
        main_model.load_state_dict(cached_state_dict)
        
        print("Starting Subset of size ",fraction," with fairness Run!")

    sub_idxs.sort()
    np.random.seed(42)
    np_sub_idxs = np.array(sub_idxs)
    np.random.shuffle(np_sub_idxs)
    loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
            transform=None),shuffle=False,batch_size=train_batch_size)

    loader_val = DataLoader(CustomDataset(x_val, y_val,transform=None),shuffle=False,\
        batch_size=batch_size)

    loader_tst = DataLoader(CustomDataset(x_tst, y_tst,transform=None),shuffle=False,\
        batch_size=batch_size)

    stop_epoch = num_epochs
    
    #for i in range(num_epochs):
    stop_count = 0
    prev_loss = 1000
    prev_loss2 = 1000
    i =0
    mul = 1
    lr_count = 0
    while (True):

        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[sub_idxs], y_trn[sub_idxs]

        temp_loss = 0.

        for batch_idx_t in list(loader_tr.batch_sampler):
            
            inputs_trn, targets_trn = loader_tr.dataset[batch_idx_t]
            inputs_trn, targets_trn = inputs_trn.to(device), targets_trn.to(device)

            main_optimizer.zero_grad()
            
            scores = main_model(inputs_trn)

            #l2_reg = 0
            #for param in main_model.parameters():
            #    l2_reg += torch.norm(param)

            l = [torch.flatten(p) for p in main_model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)

            #state_orig = copy.deepcopy(main_optimizer.state)
            
            '''alpha_extend = torch.repeat_interleave(alphas,val_size, dim=0)
            val_scores = main_model(x_val_combined)
            constraint = criterion(val_scores, y_val_combined) - delta_extend
            multiplier = torch.dot(alpha_extend,constraint)'''

            '''constraint = torch.zeros(len(x_val_list))
            for j in range(len(x_val_list)):
                
                inputs_j, targets_j = x_val_list[j], y_val_list[j]
                scores_j = main_model(inputs_j)
                constraint[j] = criterion(scores_j, targets_j) - deltas[j]'''

            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(device), targets.to(device)
            
                val_out = main_model(inputs)
                constraint += criterion(val_out, targets)            
            
            constraint /= len(loader_val.batch_sampler)
            #constraint = constraint - deltas
            multiplier = alphas*(constraint - deltas) #torch.dot(alphas,constraint)

            loss = criterion(scores, targets_trn) + reg_lambda*l2_reg*len(batch_idx_t) #+ multiplier #
            temp_loss += loss.item()
            loss.backward()

            if i % print_every == 0:  
                print(criterion(scores, targets_trn) , reg_lambda*l2_reg*len(batch_idx_t) ,multiplier)

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()
            #main_optimizer.param_groups[1]['lr'] = learning_rate/2
            
            '''for param in main_model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True'''

            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(device), targets.to(device)
            
                val_out = main_model(inputs)
                constraint += criterion(val_out, targets)
            
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - deltas
            multiplier = -1.0*alphas*constraint#torch.dot(-1.0*alphas ,constraint)

            #print(alphas,constraint)
            dual_optimizer.zero_grad()

            #main_optimizer.state = state_orig
            multiplier.backward()
            dual_optimizer.step()
            #print(main_optimizer.param_groups)
            #scheduler.step()

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            alphas.requires_grad = True
            #print(alphas)

            '''for param in main_model.parameters():
                param.requires_grad = True'''

        #print(alphas,constraint)

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            print(prev_loss,temp_loss,constraint*alphas)
            #print(main_optimizer.state)#.keys())
            #print(alphas,constraint)
            #print(criterion(scores, targets) , reg_lambda*l2_reg*len(idxs) ,multiplier)
            #print(main_optimizer.param_groups)#[0]['lr'])


        if ((i + 1) % select_every == 0) and func_name not in ['Random']:

            cached_state_dict = copy.deepcopy(main_model.state_dict())
            clone_dict = copy.deepcopy(cached_state_dict)

            alpha_orig = copy.deepcopy(alphas)

            alpha_orig.requires_grad = False
            alpha_orig = alpha_orig*((constraint >0).float())
            alpha_orig.requires_grad = True

            if func_name == 'Fair_subset':

                d_sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    train_batch_size)

                print(d_sub_idxs[:10])

                '''clone_dict = copy.deepcopy(cached_state_dict)
                alpha_orig = copy.deepcopy(alphas)

                sub_idxs = fsubset.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    train_batch_size)
                print(sub_idxs[:10])'''

                '''new_ele = set(d_sub_idxs).difference(set(sub_idxs))
                print(len(new_ele),0.1*bud)

                if len(new_ele) > 0.1*bud:
                    main_optimizer = torch.optim.Adam([
                    {'params': main_model.parameters()}], lr=max(main_optimizer.param_groups[0]['lr'],\
                        0.001))
                    
                    dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=learning_rate)

                    mul=1
                    stop_count = 0
                    lr_count = 0'''
                
                sub_idxs = d_sub_idxs

                sub_idxs.sort()
                np.random.seed(42)
                np_sub_idxs = np.array(sub_idxs)
                np.random.shuffle(np_sub_idxs)
                loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
                        transform=None),shuffle=False,batch_size=train_batch_size)

            main_model.load_state_dict(cached_state_dict)

        if abs(prev_loss - temp_loss) <= 1*mul or abs(temp_loss - prev_loss2) <= 1*mul:
            #print(main_optimizer.param_groups[0]['lr'])
            #print('lr',i)
            lr_count += 1
            if lr_count == 10:
                print(i,"Reduced",mul)
                #print(prev_loss,temp_loss,alphas)
                scheduler.step()
                mul/=10
                lr_count = 0
        else:
            lr_count = 0
        
        if (abs(prev_loss - temp_loss) <= 1e-5 or abs(temp_loss - prev_loss2) <= 1e-5) and\
             stop_count >= 10:
            print(i,prev_loss,temp_loss,constraint)
            break 
        elif abs(prev_loss - temp_loss) <= 1e-5 or abs(temp_loss - prev_loss2) <= 1e-5:
            #print(prev_loss,temp_loss)
            stop_count += 1
        else:
            stop_count = 0

        if i>=2000:
            break
        
        prev_loss2 = prev_loss
        prev_loss = temp_loss
        i +=1
        
    #print(constraint)
    #print(alphas)
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

    return [val_loss,test_loss,stop_epoch]#[val_accuracy, val_classes, tst_accuracy, tst_classes]
