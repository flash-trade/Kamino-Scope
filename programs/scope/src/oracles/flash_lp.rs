use std::ops::Deref;

use anchor_lang::prelude::*;
use anchor_spl::token::spl_token::state::Mint;
use decimal_wad::decimal::Decimal;
pub use flash_perp_itf as perpetuals;
pub use perpetuals::utils::{check_mint_pk, get_mint_pk};
use perpetuals::Custody;
use solana_program::program_pack::Pack;

use crate::{
    scope_chain::get_price_from_chain,
    utils::{account_deserialize, math::ten_pow},
    DatedPrice, MintToScopeChain, MintsToScopeChains, OraclePrices, Price, Result, ScopeError,
};
pub const POOL_VALUE_SCALE_DECIMALS: u8 = 6;

/// Gives the hypothetical price of 1 FLP token in USD
///
/// Uses the AUM of the pool and the supply of the FLP token to compute the price
pub fn get_price_no_recompute<'a, 'b>(
    flash_pool_acc: &AccountInfo,
    clock: &Clock,
    extra_accounts: &mut impl Iterator<Item = &'b AccountInfo<'a>>,
) -> Result<DatedPrice>
where
    'a: 'b,
{
    let flash_pool_pk = flash_pool_acc.key;
    let flash_pool: perpetuals::Pool = account_deserialize(flash_pool_acc)?;

    let mint_acc = extra_accounts
        .next()
        .ok_or(ScopeError::AccountsAndTokenMismatch)?;

    check_mint_pk(flash_pool_pk, mint_acc.key, flash_pool.lp_mint_bump)
        .map_err(|_| ScopeError::UnexpectedAccount)?;

    let mint = {
        let mint_borrow = mint_acc.data.borrow();
        Mint::unpack(&mint_borrow)
    }?;

    let lp_value = flash_pool.aum_usd;
    let lp_token_supply = mint.supply;

    // This is a sanity check to make sure the mint is configured as expected
    // This allows to just divide the two values to get the price
    require_eq!(mint.decimals, POOL_VALUE_SCALE_DECIMALS);

    let price_dec = Decimal::from(lp_value) / lp_token_supply;
    let dated_price = DatedPrice {
        price: price_dec.into(),
        // TODO: find a way to get the last update time
        last_updated_slot: clock.slot,
        unix_timestamp: u64::try_from(clock.unix_timestamp).unwrap(),
        ..Default::default()
    };

    Ok(dated_price)
}

pub fn validate_flp_pool(account: &Option<AccountInfo>) -> Result<()> {
    let Some(account) = account else {
        msg!("No FLP pool account provided");
        return err!(ScopeError::PriceNotValid);
    };
    let _flp_pool: perpetuals::Pool = account_deserialize(account)?;
    Ok(())
}

/// Get the price of 1 FLP token in USD
///
/// This function recompute the AUM of the pool from the custodies and the oracles
/// Required extra accounts:
/// - Mint of the FLP token
/// - All custodies of the pool
/// - All oracles of the pool (from the custodies)
/// - All markets of the pool 
pub fn get_price_recomputed<'a, 'b>(
    flash_pool_acc: &AccountInfo<'a>,
    clock: &Clock,
    extra_accounts: &mut impl Iterator<Item = &'b AccountInfo<'a>>,
) -> Result<DatedPrice>
where
    'a: 'b,
{
    // 1. Get accounts
    let flash_pool_pk = flash_pool_acc.key;
    let flash_pool: perpetuals::Pool = account_deserialize(flash_pool_acc)?;

    let mint_acc = extra_accounts
        .next()
        .ok_or(ScopeError::AccountsAndTokenMismatch)?;

    // Get custodies and oracles
    let num_custodies = flash_pool.custodies.len();
    let num_markets = flash_pool.markets.len();

    // Note: we take all the needed accounts before any check to leave the iterator in a consistent state
    // (otherwise, we could break the next price computation)
    let custodies_accs = extra_accounts.take(num_custodies).collect::<Vec<_>>();
    require!(
        custodies_accs.len() == num_custodies,
        ScopeError::AccountsAndTokenMismatch
    );

    let oracles_accs = extra_accounts.take(num_custodies).collect::<Vec<_>>();
    require!(
        oracles_accs.len() == num_custodies,
        ScopeError::AccountsAndTokenMismatch
    );

    let market_accs = extra_accounts.take(num_markets).collect::<Vec<_>>();
    require!(
        market_accs.len() == num_markets,
        ScopeError::AccountsAndTokenMismatch
    );

    // 2. Check accounts
    check_accounts(flash_pool_pk, &flash_pool, mint_acc, &custodies_accs, &market_accs).map_err(|e| {
        msg!("Error while checking accounts: {:?}", e);
        e
    })?;


    // Check of oracles will be done in the next step while deserializing custodies
    // (avoid double iteration or keeping custodies in memory)

    // 3. Get mint supply

    let lp_token_supply = get_lp_token_supply(mint_acc).map_err(|e| {
        msg!("Error while getting mint supply: {:?}", e);
        e
    })?;

    // 4. Compute AUM and prices

    let custodies_and_prices_iter = custodies_accs.into_iter().zip(oracles_accs);
    let aum_and_price_getter = |(custody_acc, oracle_acc): (&AccountInfo, &AccountInfo),
                              clock: &Clock|
     -> Result<CustodyAumResult> {
        let custody: Custody = account_deserialize(custody_acc)?;
        require!(
            custody.oracle.oracle_type == perpetuals::OracleType::Pyth,
            ScopeError::UnexpectedFlpConfiguration
        );
        require_keys_eq!(
            custody.oracle.ext_oracle_account,
            *oracle_acc.key,
            ScopeError::UnexpectedAccount
        );
        let dated_price = super::pyth::get_price(oracle_acc, clock)?;
        Ok(CustodyAumResult {
            token_amount_usd: asset_amount_to_usd(&dated_price.price, custody.assets.owned, custody.decimals),
            dated_price,
        })
    };

    let compounding_factor = Decimal::from(flash_pool.compounding_stats.active_amount as u128)
        / flash_pool.compounding_stats.total_supply;

    compute_price_from_custodies_and_prices(
        lp_token_supply,
        clock,
        custodies_and_prices_iter,
        aum_and_price_getter,
        market_accs,
        compounding_factor,
    )
    .map_err(|e| {
        msg!(
            "Error while computing price from custodies and prices: {:?}",
            e
        );
        e
    })
}

/// Get the price of 1 FLP token in USD using a scope mapping
///
/// This function recompute the AUM of the pool from the custodies and scope prices
///
/// Required extra accounts:
/// - Mint of the FLP token
/// - The scope mint to price mapping (It must be built with the same mints and order than the custodies)
/// - All custodies of the pool
/// - All markets of the pool 
pub fn get_price_recomputed_scope<'a, 'b>(
    entry_id: usize,
    flash_pool_acc: &AccountInfo<'a>,
    clock: &Clock,
    oracle_prices_pk: &Pubkey,
    oracle_prices: &OraclePrices,
    extra_accounts: &mut impl Iterator<Item = &'b AccountInfo<'a>>,
) -> Result<DatedPrice>
where
    'a: 'b,
{
    // 1. Get accounts
    let flash_pool_pk = flash_pool_acc.key;
    let flash_pool: perpetuals::Pool = account_deserialize(flash_pool_acc)?;

    let mint_acc = extra_accounts
        .next()
        .ok_or(ScopeError::AccountsAndTokenMismatch)?;

    // Get mint to price map
    let mint_to_price_map_acc_info = extra_accounts
        .next()
        .ok_or(ScopeError::AccountsAndTokenMismatch)?;
    let mint_to_price_map_acc =
        Account::<MintsToScopeChains>::try_from(mint_to_price_map_acc_info)?;
    let mint_to_price_map = mint_to_price_map_acc.deref();

    // Get custodies
    let num_custodies = flash_pool.custodies.len();

    // Note: we take all the needed accounts before any check to leave the iterator in a consistent state
    // (otherwise, we could break the next price computation)
    let custodies_accs = extra_accounts.take(num_custodies).collect::<Vec<_>>();
    require_eq!(
        custodies_accs.len(),
        num_custodies,
        ScopeError::AccountsAndTokenMismatch
    );

    require_gte!(mint_to_price_map.mapping.len(), num_custodies);

    // Get markets
    let num_markets = flash_pool.markets.len();

    let market_accs = extra_accounts.take(num_markets).collect::<Vec<_>>();
    require_eq!(
        market_accs.len(),
        num_markets,
        ScopeError::AccountsAndTokenMismatch
    );

    // 2. Check accounts
    check_accounts(flash_pool_pk, &flash_pool, mint_acc, &custodies_accs, &market_accs).map_err(|e| {
        msg!("Error while checking accounts: {:?}", e);
        e
    })?;

    require_keys_eq!(
        *oracle_prices_pk,
        mint_to_price_map.oracle_prices,
        ScopeError::UnexpectedAccount
    );

    require_keys_eq!(
        *flash_pool_pk,
        mint_to_price_map.seed_pk,
        ScopeError::UnexpectedAccount
    );

    require_eq!(
        u64::try_from(entry_id).unwrap(),
        mint_to_price_map.seed_id,
        ScopeError::UnexpectedAccount
    );
    // That the price mints matches the will be done in the next step while deserializing custodies
    // (avoid double iteration or keeping custodies in memory)

    // 3. Get mint supply

    let lp_token_supply = get_lp_token_supply(mint_acc).map_err(|e| {
        msg!("Error while getting mint supply: {:?}", e);
        e
    })?;

    

    // 4. Compute AUM and prices

    let custodies_and_prices_iter = custodies_accs
        .into_iter()
        .zip(mint_to_price_map.mapping.iter());
    let aum_and_price_getter = |(custody_acc, mint_to_chain): (&AccountInfo, &MintToScopeChain),
                              _clock: &Clock|
     -> Result<CustodyAumResult> {
        let custody: Custody = account_deserialize(custody_acc)?;
        require_keys_eq!(
            custody.mint,
            mint_to_chain.mint,
            ScopeError::UnexpectedAccount
        );
        let dated_price =
            get_price_from_chain(oracle_prices, &mint_to_chain.scope_chain).map_err(|e| {
                msg!("Error while getting price from scope chain: {:?}", e);
                ScopeError::BadScopeChainOrPrices
            })?;
        Ok(CustodyAumResult {
            token_amount_usd: asset_amount_to_usd(&dated_price.price, custody.assets.owned, custody.decimals),
            dated_price,
        })
    };

    let compounding_factor = Decimal::from(flash_pool.compounding_stats.active_amount as u128)
        / flash_pool.compounding_stats.total_supply;

    let price = compute_price_from_custodies_and_prices(
        lp_token_supply,
        clock,
        custodies_and_prices_iter,
        aum_and_price_getter,
        market_accs,
        compounding_factor,
    )
    .map_err(|e| {
        msg!(
            "Error while computing price from custodies and prices: {:?}",
            e
        );
        e
    })?;

    Ok(price)
}

fn compute_price_from_custodies_and_prices<T>(
    lp_token_supply: u64,
    clock: &Clock,
    custodies_and_prices_iter: impl Iterator<Item = T>,
    aum_and_price_getter: impl Fn(T, &Clock) -> Result<CustodyAumResult>,
    market_accs: Vec<&AccountInfo>,
    compounding_factor: Decimal,
) -> Result<DatedPrice> {
    let mut oldest_price_ts: u64 = clock.unix_timestamp.try_into().unwrap();
    let mut oldest_price_slot: u64 = clock.slot;

    let lp_value: u128 = {
        let mut pool_amount_usd: u128 = 0;

        let mut prices = Box::new(Vec::new());

        for custody_and_price in custodies_and_prices_iter {
            // Compute custody AUM
            let custody_r = aum_and_price_getter(custody_and_price, clock)?;
            prices.push(custody_r.dated_price.price);
            pool_amount_usd += custody_r.token_amount_usd;

            // Update oldest price
            if custody_r.dated_price.unix_timestamp < oldest_price_ts {
                oldest_price_ts = custody_r.dated_price.unix_timestamp;
                oldest_price_slot = custody_r.dated_price.unix_timestamp;
            }
        }

        // Compute unsettled pnl agasint the pool
        for market_acc in market_accs {
            let market = account_deserialize::<perpetuals::Market>(market_acc)?;
            if market.collective_position.size_amount == 0 {
                continue;
            }
            let collective_entry_price = Price {
                value: market.collective_position.average_entry_price.price,
                exp: market.collective_position.average_entry_price.exponent.abs().try_into().unwrap(),
            };
            let mark_price = prices[market.target_custody_id as usize];
            let collective_entry_price = collective_entry_price.scale_to_exponent(mark_price.exp)?;
            if (market.side == perpetuals::Side::Long && collective_entry_price.value < mark_price.value) ||
                (market.side == perpetuals::Side::Short && collective_entry_price.value > mark_price.value) {
                let price_diff = if market.side == perpetuals::Side::Long {
                    Price{
                        value: mark_price.value - collective_entry_price.value,
                        exp: mark_price.exp,
                    }
                } else {
                    Price{
                        value: collective_entry_price.value - mark_price.value,
                        exp: mark_price.exp,
                    }
                };
                pool_amount_usd -= asset_amount_to_usd(&price_diff, market.collective_position.size_amount, market.collective_position.size_decimals);
            } else {
                let price_diff = if market.side == perpetuals::Side::Long {
                    Price{
                        value: collective_entry_price.value - mark_price.value,
                        exp: mark_price.exp,
                    }
                } else {
                    Price{
                        value: mark_price.value - collective_entry_price.value,
                        exp: mark_price.exp,
                    }
                };
                pool_amount_usd += asset_amount_to_usd(&price_diff.try_into().unwrap(), market.collective_position.size_amount, market.collective_position.size_decimals);
            }
        }

        pool_amount_usd
    };

    // 5. Compute underlying staked lp price (sFLP)
    let sflp_price_dec = Decimal::from(lp_value) / lp_token_supply;

    // 6. Compute the price of compounding lp token (FLP)
    let price_dec = sflp_price_dec * compounding_factor;

    let dated_price = DatedPrice {
        price: price_dec.into(),
        last_updated_slot: oldest_price_slot,
        unix_timestamp: oldest_price_ts,
        ..Default::default()
    };

    Ok(dated_price)
}

fn check_accounts(
    flash_pool_pk: &Pubkey,
    flash_pool: &perpetuals::Pool,
    mint_acc: &AccountInfo,
    custodies_accs: &[&AccountInfo],
    market_accs: &[&AccountInfo],
) -> Result<()> {
    check_mint_pk(flash_pool_pk, mint_acc.key, flash_pool.lp_mint_bump)
        .map_err(|_| error!(ScopeError::UnexpectedAccount))?;

    for (expected_custody_pk, custody_acc) in flash_pool.custodies.iter().zip(custodies_accs.iter()) {
        require_keys_eq!(
            *expected_custody_pk,
            *custody_acc.key,
            ScopeError::UnexpectedAccount
        );
    }

    for (expected_market_pk, market_acc) in flash_pool.markets.iter().zip(market_accs.iter()) {
        require_keys_eq!(
            *expected_market_pk,
            *market_acc.key,
            ScopeError::UnexpectedAccount
        );
    }

    Ok(())
}

fn get_lp_token_supply(mint_acc: &AccountInfo) -> Result<u64> {
    let mint_borrow = mint_acc.data.borrow();
    let mint = Mint::unpack(&mint_borrow)?;

    // This is a sanity check to make sure the mint is configured as expected
    // This allows to just divide aum by the supply to get the price
    require_eq!(mint.decimals, POOL_VALUE_SCALE_DECIMALS);

    Ok(mint.supply)
}

struct CustodyAumResult {
    pub token_amount_usd: u128,
    pub dated_price: DatedPrice,
}

/// Return the value of the number of tokens in USD scaled by `POOL_VALUE_SCALE_DECIMALS` decimals
fn asset_amount_to_usd(price: &Price, token_amount: u64, token_decimals: u8) -> u128 {
    let price_value: u128 = price.value.into();
    let token_amount: u128 = token_amount.into();
    let token_decimals: u8 = token_decimals;
    let price_decimals: u8 = price.exp.try_into().unwrap();

    // price * 10^(-price_decimals) * token_amount * 10^(-token_decimals) * 10^POOL_VALUE_SCALE_DECIMALS
    if price_decimals + token_decimals > POOL_VALUE_SCALE_DECIMALS {
        let diff = price_decimals + token_decimals - POOL_VALUE_SCALE_DECIMALS;
        let nom = price_value * token_amount;
        let denom = ten_pow(diff);

        nom / denom
    } else {
        let diff = POOL_VALUE_SCALE_DECIMALS - (price_decimals + token_decimals);
        price_value * token_amount * ten_pow(diff)
    }
}


