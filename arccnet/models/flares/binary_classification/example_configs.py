# example_configs.py
"""
Example configurations for different loss functions in binary classification.
Copy the desired configuration to config.py to use it.
"""

# =============================================================================
# Configuration 1: Standard Binary Cross Entropy Loss
# =============================================================================
"""
# Use this for balanced datasets or when you want simple, standard training
LOSS_FUNCTION = "bce"
USE_CLASS_WEIGHTS = False  # Not used for standard BCE
FOCAL_ALPHA = 0.25  # Not used for standard BCE
FOCAL_GAMMA = 2.0   # Not used for standard BCE
"""

# =============================================================================
# Configuration 2: Weighted Binary Cross Entropy Loss
# =============================================================================
"""
# Use this for imbalanced datasets where you want to weight the minority class
LOSS_FUNCTION = "weighted_bce"
USE_CLASS_WEIGHTS = True  # This will use computed class weights
FOCAL_ALPHA = 0.25  # Not used for weighted BCE
FOCAL_GAMMA = 2.0   # Not used for weighted BCE
"""

# =============================================================================
# Configuration 3: Focal Loss - Conservative (less aggressive)
# =============================================================================
"""
# Use this for moderately imbalanced datasets
LOSS_FUNCTION = "focal"
USE_CLASS_WEIGHTS = False  # Not used for focal loss
FOCAL_ALPHA = 0.25  # Standard weight for rare class
FOCAL_GAMMA = 1.0   # Lower gamma = less focus on hard examples
"""

# =============================================================================
# Configuration 4: Focal Loss - Standard (recommended starting point)
# =============================================================================
"""
# Use this for imbalanced datasets (most common configuration)
LOSS_FUNCTION = "focal"
USE_CLASS_WEIGHTS = False  # Not used for focal loss
FOCAL_ALPHA = 0.25  # Standard weight for rare class
FOCAL_GAMMA = 2.0   # Standard focusing parameter
"""

# =============================================================================
# Configuration 5: Focal Loss - Aggressive (for highly imbalanced datasets)
# =============================================================================
"""
# Use this for highly imbalanced datasets with many easy examples
LOSS_FUNCTION = "focal"
USE_CLASS_WEIGHTS = False  # Not used for focal loss
FOCAL_ALPHA = 0.25  # Standard weight for rare class
FOCAL_GAMMA = 5.0   # Higher gamma = more focus on hard examples
"""

# =============================================================================
# Configuration 6: Focal Loss - More weight to rare class
# =============================================================================
"""
# Use this when you want to give more importance to the minority class
LOSS_FUNCTION = "focal"
USE_CLASS_WEIGHTS = False  # Not used for focal loss
FOCAL_ALPHA = 0.75  # Higher alpha = more weight to rare class
FOCAL_GAMMA = 2.0   # Standard focusing parameter
"""

# =============================================================================
# Guidelines for choosing loss functions:
# =============================================================================
"""
1. **Standard BCE (bce)**:
   - Use for balanced datasets
   - Simple and fast
   - Good baseline

2. **Weighted BCE (weighted_bce)**:
   - Use for imbalanced datasets where you want simple class weighting
   - Automatically computes weights based on class frequencies
   - Less complex than focal loss

3. **Focal Loss (focal)**:
   - Use for imbalanced datasets, especially with many easy examples
   - Focuses training on hard examples
   - More sophisticated than simple class weighting
   - Start with alpha=0.25, gamma=2.0 and adjust based on results

   **Focal Loss Parameter Guidelines:**
   - **Alpha (α)**: Weight for rare class
     * 0.25 = standard, more weight to common class
     * 0.5 = equal weight
     * 0.75 = more weight to rare class

   - **Gamma (γ)**: Focusing parameter
     * 0 = equivalent to standard cross-entropy
     * 1.0 = mild focusing on hard examples
     * 2.0 = standard focusing (recommended starting point)
     * 5.0 = aggressive focusing on hard examples

**Performance Tips:**
- For solar flare detection (typically highly imbalanced), start with focal loss
- If focal loss doesn't converge well, try weighted BCE
- Monitor both precision and recall, not just accuracy
- Use F1-score as your main metric for imbalanced datasets
"""
