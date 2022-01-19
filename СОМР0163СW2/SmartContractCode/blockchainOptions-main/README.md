# blockchainOptions
Option trading as a decentralised entity on the blockchain

**dXdY Instructions**
1. Load the code in remix
2. Compile creator.sol
3. Deploy creator.sol and make sure to choose Injected Web3 for Environment and creator.sol for Contract

![image](https://user-images.githubusercontent.com/15021790/144902831-12ef5d3a-6457-4272-9231-a89ae13af385.png)

4. Copy the computed hash 

![image](https://user-images.githubusercontent.com/15021790/144903122-d2708221-a105-4721-857d-0cd9a85ba66f.png)

5. Paste it in the etherscan.io of your chosen metamast network. Mine is ropsten.etherscan.io because I can get test ether easier
6. Open the Contract address

![image](https://user-images.githubusercontent.com/15021790/144903382-4bd50054-167a-486a-b4ab-ec101b9f7ec2.png)

7. Verify it if needed. This can be done by flattening the files from remix with a flattener plugin

![image](https://user-images.githubusercontent.com/15021790/144903577-1caca09a-f305-4eac-993a-ebbf49ae7cc4.png)

8. Go to Write Contract and populate the createCoveredOption. Could be something similar to my example. You can get the timestamp from https://www.epochconverter.com/

![image](https://user-images.githubusercontent.com/15021790/144903923-2aac5a7e-aa23-46af-909c-e0c72b596157.png)

9. Put the same values in getCoveredOption to retrieve the address of your newly created option

![image](https://user-images.githubusercontent.com/15021790/144904186-d0069407-2c11-4f17-ad84-069ee69c3ab5.png)

10. Verify it if needed. After you've verified it once you won't need to do it again because it will be the same ByteCode. Use the flattener again for options.sol and remember to add the constructor arguments in the verificaiton. You can do that with the help from https://abi.hashex.org/

![image](https://user-images.githubusercontent.com/15021790/144904437-005948c3-4727-463d-9f05-a903fe5b5c13.png)

11. If you want to execute buyOption from the new option contract, you need to pick buyer and writer. Then, from the buyer copy the option address and give allowance to it so that your underlying token can be transferred to the option contract.

![image](https://user-images.githubusercontent.com/15021790/144904831-fdec2f3e-a04d-44bc-932f-c35553e7d3be.png)

12. After you've given allowance from your ERC20 token, go back to the option contract and login with the buyer.

13. Populate the buyOption with the required ether and writer address. Note that the ether is specified in wei and you can use https://eth-converter.com/ to convert it correctly

![image](https://user-images.githubusercontent.com/15021790/144905445-623ac374-cad6-480c-9641-699d44d5052f.png)

14. To exercise the option just specify the strike price and the call will succeed if the exact strike price is added, option hasn't expired and you have bought an option

![image](https://user-images.githubusercontent.com/15021790/144905882-7db70197-ed0f-4b69-81f3-6d375b0a81bf.png)

15. The funds from the exercised options can only be retrieved when the option has expired. When that's done you just specify the writer's address in retrieveExpiredFunds

![image](https://user-images.githubusercontent.com/15021790/144906124-6c36a62e-e73e-4c19-8eaf-4b646065b61a.png)

16. You can see the state of all options getOptionProperties


####################################################################


**Chainlink Instructions**

Much easier than dYdX. Has a bit more properties like tokenPrice, but they are not that relevant to the actual options operations.
1. Publish the contract
2. Login with writer to write the option
3. Before writing the option make sure that you've given allowance to the option address to transferFrom your underlying ERC20 token
4. login with buyer to buy the option. The id of the option is just its index in the list.
5. Exercise the option and both writer and buyer will get their funds immediately




