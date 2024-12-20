#![allow(clippy::result_large_err)]

pub mod states;
pub mod utils;

use anchor_lang::prelude::*;
pub use states::*;

declare_id!("FLASH6Lo6h3iasJKWDs2F8TkW2UKf3s15C8PMGuVfgBn");

pub const PRICE_DECIMALS: u8 = 6;

#[program]
pub mod perpetuals {
    use super::*;

    #[allow(unused_variables)]
    pub fn get_assets_under_management(
        ctx: Context<GetAssetsUnderManagement>,
    ) -> Result<u128> {
        // We only need the interface, not the actual implementation here.
        unimplemented!("flash-perp-itf is just an interface")
    }
}


#[derive(Accounts)]
pub struct GetAssetsUnderManagement<'info> {

    /// CHECK: don't care this is just an interface
    #[account()]
    pub perpetuals: Box<Account<'info, Perpetuals>>,

    /// CHECK: don't care this is just an interface
    #[account()]
    pub pool: Box<Account<'info, Pool>>,

    // remaining accounts:
    //   pool.custodies.len() custody accounts (read-only, unsigned)
    //   pool.custodies.len() custody oracles (read-only, unsigned)
    //   pool.markets.len() market accounts (read-only, unsigned)
}
