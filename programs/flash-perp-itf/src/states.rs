use anchor_lang::prelude::*;

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct Permissions {
    pub allow_swap: bool,
    pub allow_add_liquidity: bool,
    pub allow_remove_liquidity: bool,
    pub allow_open_position: bool,
    pub allow_close_position: bool,
    pub allow_collateral_withdrawal: bool,
    pub allow_size_change: bool,
    pub allow_liquidation: bool,
    pub allow_flp_staking: bool,
    pub allow_fee_distribution: bool,
    pub allow_ungated_trading: bool,
    pub allow_fee_discounts: bool,
    pub allow_referral_rebates: bool,
}

// multiplier have implied RATE_DECIMALS
#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct VoltageMultiplier {
    pub volume: u64,
    pub rewards: u64,
    pub rebates: u64,
}

#[account]
#[derive(Default, Debug)]
pub struct Perpetuals {
    pub permissions: Permissions,
    pub pools: Vec<Pubkey>,
    pub collections: Vec<Pubkey>,
    
    pub voltage_multiplier: VoltageMultiplier,
    // discounts have implied RATE_DECIMALS
    pub trading_discount: [u64; 6], 
    pub referral_rebate: [u64; 6],
    pub referral_discount: u64,
    pub inception_time: i64,

    pub transfer_authority_bump: u8,
    pub perpetuals_bump: u8,
    pub trade_limit: u16,
    pub rebate_limit_usd: u32
}

impl Perpetuals {
    pub const LEN: usize = 8 + std::mem::size_of::<Perpetuals>();
    pub const BPS_DECIMALS: u8 = 4;
    pub const BPS_POWER: u128 = 10u64.pow(Self::BPS_DECIMALS as u32) as u128;
    pub const USD_DECIMALS: u8 = 6;
    pub const LP_DECIMALS: u8 = Self::USD_DECIMALS;
    pub const LP_POWER: u128 = 10u64.pow(Self::LP_DECIMALS as u32) as u128;
    pub const RATE_DECIMALS: u8 = 9;
    pub const RATE_POWER: u128 = 10u64.pow(Self::RATE_DECIMALS as u32) as u128;
    pub const DAY_SECONDS: i64 = 86400;
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Debug)]
pub enum OracleType {
    None,
    Custom,
    Pyth,
}

impl Default for OracleType {
    fn default() -> Self {
        Self::Custom
    }
}

#[derive(Copy, Clone, Eq, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct OraclePrice {
    pub price: u64,
    pub exponent: i32,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct OracleParams {
    pub int_oracle_account: Pubkey,
    pub ext_oracle_account: Pubkey,
    pub oracle_type: OracleType,
    pub max_divergence_bps: u64,
    pub max_conf_bps: u64,
    pub max_price_age_sec: u64, 
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct TokenRatios {
    pub target: u64,
    pub min: u64,
    pub max: u64,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct CompoundingStats {
    pub active_amount: u64,
    pub total_supply: u64,
    pub reward_snapshot: u128,
    pub fee_share_bps: u64,
    pub last_compound_time: i64,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct StakeStats {
    pub pending_activation: u64,
    pub active_amount: u64,
    pub pending_deactivation: u64,
    pub deactivated_amount: u64,
}

#[account]
#[derive(Default, Debug)]
pub struct Pool {
    pub name: String,
    pub permissions: Permissions,
    pub inception_time: i64,
    pub lp_mint: Pubkey,
    pub oracle_authority: Pubkey,
    pub staked_lp_vault: Pubkey,
    pub reward_custody: Pubkey,
    pub custodies: Vec<Pubkey>,
    pub ratios: Vec<TokenRatios>,
    pub markets: Vec<Pubkey>,
    pub max_aum_usd: u128,
    pub aum_usd: u128,
    pub total_staked: StakeStats,
    pub staking_fee_share_bps: u64,
    pub bump: u8,
    pub lp_mint_bump: u8,
    pub staked_lp_vault_bump: u8,
    pub vp_volume_factor: u8,
    pub padding: [u8; 4],
    pub staking_fee_boost_bps: [u64; 6], 
    pub compounding_mint: Pubkey,
    pub compounding_lp_vault: Pubkey,
    pub compounding_stats: CompoundingStats,
    pub compounding_mint_bump: u8,
    pub compounding_lp_vault_bump: u8,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Debug)]
pub enum FeesMode {
    Fixed,
    Linear,
}

impl Default for FeesMode {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct Fees {
    pub mode: FeesMode,
    // fees have implied RATE_DECIMALS
    pub swap_in: RatioFees,
    pub swap_out: RatioFees,
    pub stable_swap_in: RatioFees,
    pub stable_swap_out: RatioFees,
    pub add_liquidity: RatioFees,
    pub remove_liquidity: RatioFees,
    pub open_position: u64,
    pub close_position: u64,
    pub remove_collateral: u64,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct RatioFees {
    pub min_fee: u64,
    pub target_fee: u64,
    pub max_fee: u64
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct Assets {
    // Collateral held in custody
    pub collateral: u64,
    // Deposited by LPs and pnl settled against the pool
    pub owned: u64,
    // Locked funds for pnl payoff
    pub locked: u64,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct FeesStats {
    // Fees accrued by the custody
    pub accrued: u128,
    // Fees distributed to the staked LPs
    pub distributed: u128,
    // Fees collected by LPs
    pub paid: u128,
    // Fees accrued per staked LP token
    pub reward_per_lp_staked: u64,
    // Protocol share of the fees
    pub protocol_fee: u64,
}



#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct PricingParams {
    pub trade_spread_min: u64, // in 100th of bps 
    pub trade_spread_max: u64, // in 100th of bps 
    pub swap_spread: u64, // BPS_DECIMALS
    pub min_initial_leverage: u64, // BPS_DECIMALS
    pub max_initial_leverage: u64, // BPS_DECIMALS
    pub max_leverage: u64, // BPS_DECIMALS
    pub min_collateral_usd: u64,
    pub delay_seconds: i64,
    pub max_utilization: u64, // BPS_DECIMALS
    pub max_position_locked_usd: u64,
    pub max_exposure_usd: u64,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct BorrowRateParams {
    // borrow rate params have implied RATE_DECIMALS decimals
    pub base_rate: u64,
    pub slope1: u64,
    pub slope2: u64,
    pub optimal_utilization: u64,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct BorrowRateState {
    // borrow rates have implied RATE_DECIMALS decimals
    pub current_rate: u64,
    pub cumulative_lock_fee: u128,
    pub last_update: i64,
}

#[account]
#[derive(Default, Debug, PartialEq)]
pub struct Custody {
    // static parameters
    pub pool: Pubkey,
    pub mint: Pubkey,
    pub token_account: Pubkey,
    pub decimals: u8,
    pub is_stable: bool,
    pub depeg_adjustment: bool,
    pub is_virtual: bool,
    pub distribute_rewards: bool,  // Flag to initialise fee distribution
    pub oracle: OracleParams,
    pub pricing: PricingParams,
    pub permissions: Permissions,
    pub fees: Fees,
    pub borrow_rate: BorrowRateParams,
    pub reward_threshold: u64,

    // dynamic variables
    pub assets: Assets,
    pub fees_stats: FeesStats,
    pub borrow_rate_state: BorrowRateState,

    // bumps for address validation
    pub bump: u8,
    pub token_account_bump: u8,

    pub size_factor_for_spread: u8,
    pub null: u8,
    pub reserved_amount: u64, // TODO: Requires Realloc and shoulkd also be checked along assets
    pub min_reserve_usd: u64,
    pub limit_price_buffer_bps: u64, // BPS_DECIMALS
    pub padding: [u8; 32],
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct MarketPermissions {
    pub allow_open_position: bool,
    pub allow_close_position: bool,
    pub allow_collateral_withdrawal: bool,
    pub allow_size_change: bool,
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Debug)]
pub enum Side {
    None,
    Long,
    Short,
}

impl Default for Side {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Copy, Clone, PartialEq, AnchorSerialize, AnchorDeserialize, Default, Debug)]
pub struct PositionStats {
    pub open_positions: u64,
    pub update_time: i64,
    pub average_entry_price: OraclePrice,
    pub size_amount: u64,
    pub size_usd: u64,
    pub locked_amount: u64,
    pub locked_usd: u64,
    pub collateral_amount: u64,
    pub collateral_usd: u64, // Only used for persistent storage
    pub unsettled_fee_usd: u64,
    pub cumulative_lock_fee_snapshot: u128,
    pub size_decimals: u8,
    pub locked_decimals: u8,
    pub collateral_decimals: u8,
}

#[account]
#[derive(Default, Debug, PartialEq)]
pub struct Market {
    pub pool: Pubkey,
    pub target_custody: Pubkey,
    pub collateral_custody: Pubkey,
    pub side: Side,
    pub correlation: bool,
    pub max_payoff_bps: u64,
    pub permissions: MarketPermissions,
    pub open_interest: u64,
    pub collective_position: PositionStats,
    pub target_custody_id: usize,
    pub collateral_custody_id: usize,
    pub bump: u8,
}

#[account]
#[derive(Default, Debug)]
pub struct Position {
    pub owner: Pubkey,
    pub market: Pubkey,
    pub delegate: Pubkey, //For later use
    pub open_time: i64,
    pub update_time: i64,
    pub entry_price: OraclePrice,
    pub size_amount: u64,
    pub size_usd: u64,
    pub locked_amount: u64,
    pub locked_usd: u64,
    pub collateral_amount: u64,
    pub collateral_usd: u64,
    pub unsettled_amount: u64, // Used for position delta accounting
    pub unsettled_fees_usd: u64,
    pub cumulative_lock_fee_snapshot: u128,
    pub take_profit_price: OraclePrice,
    pub stop_loss_price: OraclePrice,
    pub size_decimals: u8,
    pub locked_decimals: u8,
    pub collateral_decimals: u8,
    pub bump: u8,
}
