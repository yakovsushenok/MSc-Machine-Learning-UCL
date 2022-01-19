// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

import "./ERC20_own.sol";

contract chainlinkOptions {    
    address payable contractAddr;
    
    //Options stored in arrays of structs
    struct option {
        address underlyingToken;    
        uint256 underlyingTokenPrice;
        uint256 strike; //Price in wei option allows buyer to purchase tokens at
        uint256 premium; //Fee in contract token that option writer charges
        uint256 expiry; //Unix timestamp of expiration time
        uint256 amount; //Amount of tokens the option contract is for
        bool exercised; //Has option been exercised
        bool canceled; //Has option been canceled
        address payable writer; //Issuer of option
        address payable buyer; //Buyer of option
    }

    option[] public tokenOpts;

    constructor() {
        contractAddr = payable(address(this));
    }

    // Allows user to write a covered call option
    // Takes which token, a strike price(USD per token w/18 decimal places), premium(same unit as token), expiration time(unix) and how many tokens the contract is for
    function writeOption(address underlyingTokenAddress, uint256 underlyingTokenPrice, uint strike, uint premium, uint expiry, uint tknAmt) public {
        // Need to authorize the contract to transfer funds on your behalf
        require(ERC20(underlyingTokenAddress).transferFrom(msg.sender, contractAddr, tknAmt), "Incorrect amount of TOKEN supplied");
        tokenOpts.push(option(underlyingTokenAddress, underlyingTokenPrice, strike, premium, expiry, tknAmt, false, false, payable(msg.sender), payable(address(0))));
    }
    
    // Purchase a call option, needs desired token, ID of option and payment
    function buyOption(uint256 ID) public payable {
        // Transfer premium payment from buyer to writer
        // Need to authorize the contract to transfer funds on your behalf
        require(tokenOpts[ID].buyer == address(0), "The option is already bought");
        require(msg.value == tokenOpts[ID].premium, "Incorrect amount of ETH sent for premium");
        tokenOpts[ID].writer.transfer(tokenOpts[ID].premium);
        tokenOpts[ID].buyer = payable(msg.sender);
    }
    
    // Exercise your call option, needs desired token, ID of option and payment
    function exercise(uint256 ID) public payable {
        // If not expired and not already exercised, allow option owner to exercise
        // To exercise, the strike value*amount equivalent paid to writer (from buyer) and amount of tokens in the contract paid to buyer
        require(tokenOpts[ID].buyer == msg.sender, "You do not own this option");
        require(!tokenOpts[ID].exercised, "Option has already been exercised");
        require(tokenOpts[ID].expiry >= block.timestamp, "Option is expired");
        uint256 exerciseVal = tokenOpts[ID].strike*tokenOpts[ID].amount;
        require(msg.value == exerciseVal, "Incorrect ETH amount sent to exercise");
        
        // Buyer exercises option, exercise cost paid to writer
        tokenOpts[ID].writer.transfer(exerciseVal);

        // Pay buyer contract amount of TOKEN
        require(ERC20(tokenOpts[ID].underlyingToken).transfer(msg.sender, tokenOpts[ID].amount), "Error: buyer was not paid");
        tokenOpts[ID].exercised = true;
    }
            
    // Allows writer to retrieve funds from an expired, non-exercised, non-canceled option
    function retrieveExpiredFunds(uint ID) public {
        require(msg.sender == tokenOpts[ID].writer, "You did not write this option");
        require(tokenOpts[ID].expiry <= block.timestamp && !tokenOpts[ID].exercised && !tokenOpts[ID].canceled, "This option is not eligible for withdraw");
        require(ERC20(tokenOpts[ID].underlyingToken).transfer(tokenOpts[ID].writer, tokenOpts[ID].amount), "Incorrect amount of LINK sent");
        tokenOpts[ID].canceled = true;
    }

    // Allows option writer to cancel and get their funds back from an unpurchased option
    function cancelOption(uint ID) public {
        require(msg.sender == tokenOpts[ID].writer, "You did not write this option");
        require(!tokenOpts[ID].canceled && tokenOpts[ID].buyer == address(0), "This option cannot be canceled");
        require(ERC20(tokenOpts[ID].underlyingToken).transfer(tokenOpts[ID].writer, tokenOpts[ID].amount), "Incorrect amount of LINK sent");
        tokenOpts[ID].canceled = true;
    }
    
    //Updates underlyingTokenPrice member of option which is publicly viewable
    function updateUnderlyingTokenPrice(uint256 ID, uint256 underlyingTokenPrice) public {
        tokenOpts[ID].underlyingTokenPrice = underlyingTokenPrice;
    }
}
