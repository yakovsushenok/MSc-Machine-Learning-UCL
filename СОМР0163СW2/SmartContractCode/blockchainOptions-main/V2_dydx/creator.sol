// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

import "./options.sol";

contract DerivativeCreator {

    // Mapping storing all token options in existence
    mapping(bytes32 => address) tokenOpts;

    /**
     * Create a new type of covered option
     * Will create a new CoveredOption smart contract and return its address
     *
     * @param  underlyingToken            The address of the underlying token used in the option
     * @param  expirationTimestamp        A timestamp indicating the expiration date of the option
     * @param  underlyingTokenStrikePrice The underlyingToken strike price
     * @return _option                    The address of the new option contract
     */
    function createCoveredOption(
        address underlyingToken,
        uint256 underlyingTokenStrikePrice,
        uint256 premium,
        uint256 expirationTimestamp
    ) public returns (address _option) {
        bytes32 optionHash = keccak256(abi.encodePacked(
            underlyingToken,
            underlyingTokenStrikePrice,
            premium,
            expirationTimestamp
        ));

        require(tokenOpts[optionHash] == address(0));

        address option = address(new DecentralisedOption(
            underlyingToken,
            underlyingTokenStrikePrice,
            premium,
            expirationTimestamp
        ));

        tokenOpts[optionHash] = option;

        return option;
    }

    /**
     * Get the address of a covered option contract. Will return the 0 address if none exists
     *
     * @param  underlyingToken            The address of the underlying token used in the option
     * @param  expirationTimestamp        A timestamp indicating the expiration date of the option
     * @param  underlyingTokenStrikePrice  The underlyingToken strike price
     * @return _option                    The address of the option contract
     */
    function getCoveredOption(
        address underlyingToken,
        uint256 underlyingTokenStrikePrice,
        uint256 premium,
        uint256 expirationTimestamp
        
    ) view public returns(address _option) {
        bytes32 optionHash = keccak256(abi.encodePacked(
            underlyingToken,
            underlyingTokenStrikePrice,
            premium,
            expirationTimestamp
        ));

        return tokenOpts[optionHash];
    }
}
