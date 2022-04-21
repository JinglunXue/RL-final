# Summary of ‘Reward-shaping-to-improve-the-performance-of-DRL-in-perishable-inventory-management’

Link to paper: https://www.ssrn.com/abstract=3804655

### Abstract:

This paper focus on the learning process and ultimate performance of DRL for perishable inventory control. The authors demonstrate how potential-based reward shaping can be used to leverage domain knowledge embedded in heuristic inventory policies, which is called 'the teacher policy'. They consider two teacher policies here: base-stock policy and a modied base-stock policy with estimates of waste (BSP-low-EW). They find that applying their approach using existing replenishment policies may not only reduce firms' replenishment costs, the increased stability may also help to gain trust in the policies obtained by black box DRL algorithms. This work use pytorch instead of keras, and  further investigate the impact of different loss function (L1 loss) , and the impact of different network structure.

### Code Structure:

* TRAINER.py -- trains object from class unshaped_dqn, shaped_b or shaped_ble

* UNSHAPED_DQN_pytorch.py -- deep Q-network without reward shaping

* SHAPED_B_pytorch.py -- deep Q-network with reward shaping with base-stock policy as teacher

* SHAPED_BLE_pytorch.py -- deep Q-network with reward shaping with BSP-low-EW as teacher

* B.py -- agent who use base-stock policy

* BLE -- agent who use BSP-low-EW policy

* ENV_TRAIN.py -- perishable inventory problem to train the DRL models

* ENV_TEST.py -- perishable inventory problem to test the DRL models with seeded demand

* EVALUATE_DRL_POLICY.py -- evaluates a trained DRL model in the test environment

**For changing to L1 loss, you can use line 60 of UNSHAPED_DQN_pytorch.py into line 61.**

### Usage:

The following line can be used to start the training for UNSHAPED_DQN, SHAPED_B and SHAPED_BLE agent:

- python TRAINER.py

The following lines can be used to get the performance of agent using base-stock policy and BSP-low-EW policy:

- python B.py

- python BLE.py

The results are writed into the excel files in the EVAL0 file in current directory.