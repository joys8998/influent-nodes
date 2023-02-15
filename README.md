# influent-nodes

Algorithm to get more "influent" nodes based on A Comprehensive Algorithm for Evaluating Node Influences in Social Networks Based on Preference Analysis and Random Walk paper available here https://www.hindawi.com/journals/complexity/2018/1528341/

1) Get Differential Expression data
2) transform them into a protein-protein interaction network
3) run the script 

Metrics used to calculate node influence
- DegreeCentrality (standard)
- BetweennessCentrality (standard)
- ClosenessCentrality (standard)
- EigenvectorCentrality (used by Google)
- CurrentFlowBetweennessCentrality (elaboration of BetweennessCentrality)
- Reachability (elaboration of DegreeCentrality, ClosenessCentrality and EigenvectorCentrality)
