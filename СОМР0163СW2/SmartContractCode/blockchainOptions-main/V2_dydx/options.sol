// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

import "./ERC20_own.sol";

contract DecentralisedOption {    
    address payable contractAddr;
    
    mapping (address => uint) private balances;

    // Mapping containing the numbers of options written by each address
    mapping(address => uint256) private writers;

    //Options stored in arrays of structs
    struct option {
        address underlyingToken;
        uint256 strike; 
        uint256 premium; //Fee in contract token that option writer charges
        uint256 expiry; //Unix timestamp of expiration time
        uint256 totalWritten; // Total number of options written
        uint256 totalExercised; // Total number of options exercised
        uint256 totalWithdrawn; // Total number of options withdrawn after expiration
        uint256 totalOptions;
        uint256 totalUnderlyingToken; // Total number of options outstanding
        uint256 totalBaseToken; // Total amount of baseToken collected on option exercise and held by this contract
    }

    option public tokenOpt;

    constructor(address _underlyingTokenAddress, uint _strike, uint _premium, uint _expiry) {
        contractAddr = payable(address(this));
        tokenOpt = option(_underlyingTokenAddress, _strike, _premium, _expiry, 0, 0, 0, 0, 0, 0);
    }
  
    //Purchase a call option, needs desired token, ID of option and payment
    function buyOption(address writer) public payable {
        // Transfer premium payment from buyer to writer
        // Need to authorize the contract to transfer funds on your behalf
        require(msg.value == tokenOpt.premium, "Incorrect amount of ETH sent for premium");
        payable(writer).transfer(tokenOpt.premium);

        require(ERC20(tokenOpt.underlyingToken).transferFrom(writer, contractAddr, 1), "Incorrect amount of TOKEN supplied");
        
        // Increment balances
        balances[msg.sender] += 1;
        writers[writer] += 1;

        // Increment totals
        tokenOpt.totalUnderlyingToken += 1;
        tokenOpt.totalOptions += 1;
        tokenOpt.totalWritten += 1;
    }
    
    // Exercise your call option, needs desired token, ID of option and payment
    function exercise() public payable {
        // If not expired and not already exercised, allow option owner to exercise
        // To exercise, the strike value*amount equivalent paid to writer (from buyer) and amount of tokens in the contract paid to buyer
        require(tokenOpt.expiry >= block.timestamp, "Option is expired");
        
        // require that buyer has at least one option address
        require(balances[msg.sender] >= 1);

        uint256 exerciseVal = tokenOpt.strike;
        require(msg.value == exerciseVal, "Incorrect ETH amount sent to exercise");
        
        // Pay buyer contract amount of TOKEN
        require(ERC20(tokenOpt.underlyingToken).transfer(msg.sender, 1), "Error: buyer was not paid");
    
        // Deduct balance
        balances[msg.sender] -= 1;
        
        // Update totals
        tokenOpt.totalExercised += 1;
        tokenOpt.totalOptions -= 1;
        tokenOpt.totalUnderlyingToken -= 1;
        tokenOpt.totalBaseToken += exerciseVal;
    }
            
    //Allows writer to retrieve funds from an expired, non-exercised, non-canceled option
    function retrieveExpiredFunds(address writer) public {
        uint256 balance = writers[writer];
        require(block.timestamp > tokenOpt.expiry, "The option has not expired");
        require(balance > 0);

        // Zero the writer's written balance
        writers[writer] = 0;

        // TODO: Rounding error in the division
        uint underlyingTokenAmount = tokenOpt.totalOptions * balance / tokenOpt.totalWritten;
        uint baseTokenAmount = tokenOpt.totalBaseToken * balance / tokenOpt.totalWritten;

        if (underlyingTokenAmount > 0) {
            require(ERC20(tokenOpt.underlyingToken).transfer(writer, underlyingTokenAmount));
            tokenOpt.totalUnderlyingToken -= underlyingTokenAmount;
        }

        if (baseTokenAmount > 0) {
            payable(writer).transfer(baseTokenAmount);
        }
        
        tokenOpt.totalWithdrawn += balance;
    }

    function writtenBy(address who) view public returns (uint256) {
        return writers[who];
    }

    function getActiveUnderlyingBalance() public view returns (uint256) {
        return tokenOpt.totalUnderlyingToken;
    }

    function getActiveBaseTokenBalance() public view returns (uint256) {
        return address(this).balance;
    }

    // ---------------------
    // ------- ERC20 -------
    // ---------------------

    mapping (address => mapping (address => uint256)) allowed;

    function transfer(
        address to,
        uint value
    ) public returns (
        bool ok
    ) {
        if (balances[msg.sender] >= value) {
            balances[msg.sender] -= value;
            balances[to] += value;
            return true;
        } else {
            return false;
        }
    }

    function transferFrom(
        address from,
        address to,
        uint value
    ) public returns (
        bool ok
    ) {
        if (balances[from] >= value && allowed[from][msg.sender] >= value) {
            balances[to] += value;
            balances[from] -= value;
            allowed[from][msg.sender] -= value;
            return true;
        } else {
            return false;
        }
    }

    function approve(address spender, uint value) public returns (bool ok) {
      allowed[msg.sender][spender] = value;
      return true;
    }

    function totalSupply() view public returns (uint supply) {
        return tokenOpt.totalOptions;
    }

    function balanceOf(address who) view public returns (uint value) {
        return balances[who];
    }

    function allowance(address owner, address spender) view public returns (uint _allowance) {
        return allowed[owner][spender];
    }
}
