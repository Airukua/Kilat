"""
Unit tests for Kilat Input & Tensor Validators.
Follows BDD structure from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch
from kilat.utils.validators import (
    validate_positive_int,
    validate_non_negative_float,
    validate_probability,
    validate_choice,
    validate_divisible,
    validate_sequence_length,
    validate_tensor_rank,
    validate_tensor_last_dim,
    validate_tensor_shape,
    validate_finite_tensor,
    validate_optional_dict,
    validate_all_finite_tensors,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_positive_int_validation():
    with Scenario("Unit: Positive Integer Validation"):
        with Given("a positive integer candidate value"):
            with When("validate_positive_int is executed on valid input 42"):
                result = validate_positive_int("test_param", 42)
                with Then("the original integer value is returned intact"):
                    assert result == 42

        with Given("a non-positive integer candidate value"):
            with When("validate_positive_int is executed on 0"):
                with Then("a ValueError is raised with clear parameter name in message"):
                    with pytest.raises(ValueError, match="test_param"):
                        validate_positive_int("test_param", 0)

            with When("validate_positive_int is executed on a negative number"):
                with And("a ValueError is raised for negative input"):
                    with pytest.raises(ValueError):
                        validate_positive_int("neg_param", -5)


def test_probability_validation():
    with Scenario("Unit: Probability Float Validation"):
        with Given("a floating point probability candidate"):
            with When("value is inside the half-open interval [0.0, 1.0)"):
                val = validate_probability("dropout", 0.1)
                with Then("the probability value is returned"):
                    assert val == 0.1

            with When("value is out of bounds (greater than or equal to 1.0 with upper_open=True)"):
                with Then("a ValueError is raised for upper bound violation"):
                    with pytest.raises(ValueError):
                        validate_probability("dropout", 1.0, upper_open=True)

            with When("value is negative"):
                with And("a ValueError is raised for negative probability"):
                    with pytest.raises(ValueError):
                        validate_probability("dropout", -0.01)


def test_choice_and_divisible_validation():
    with Scenario("Unit: Categorical Choice & Divisibility Validation"):
        with Given("a set of valid choices"):
            valid_choices = ["dense", "moe"]
            with When("a valid choice is provided"):
                res = validate_choice("ffn_mode", "dense", valid_choices)
                with Then("the choice string is returned"):
                    assert res == "dense"

            with When("an invalid choice is provided"):
                with Then("a ValueError is raised listing allowed options"):
                    with pytest.raises(ValueError):
                        validate_choice("ffn_mode", "invalid_mode", valid_choices)

        with Given("integers tested for divisibility"):
            with When("numerator is evenly divisible by denominator"):
                div_res = validate_divisible("hidden_size", 768, 12)
                with Then("the value is returned"):
                    assert div_res == 768

            with When("numerator is not divisible by denominator"):
                with And("a ValueError is raised for non-divisible numerator"):
                    with pytest.raises(ValueError):
                        validate_divisible("hidden_size", 768, 13)


def test_tensor_shape_and_finite_validation():
    with Scenario("Unit: Tensor Rank, Shape and Numerical Finiteness"):
        with Given("a valid 3D PyTorch float tensor"):
            t = torch.randn(2, 8, 64)
            with When("validating rank matches expected 3 dimensions"):
                ranked = validate_tensor_rank(t, 3, "embeddings")
                with Then("the tensor is returned unaltered"):
                    assert ranked.shape == (2, 8, 64)

            with When("validating trailing dimension matches 64"):
                last_dim = validate_tensor_last_dim(t, 64, "embeddings")
                with Then("validation succeeds"):
                    assert last_dim.size(-1) == 64

            with When("validating exact full shape (2, 8, 64)"):
                shaped = validate_tensor_shape(t, (2, 8, 64), "embeddings")
                with Then("the tensor shape matches"):
                    assert tuple(shaped.shape) == (2, 8, 64)

            with When("validating numerical finiteness on clean tensor"):
                finite = validate_finite_tensor(t, "clean_tensor")
                with Then("finite tensor passes check without error"):
                    assert finite is t

        with Given("a corrupted tensor containing NaN values"):
            nan_tensor = torch.tensor([1.0, float("nan"), 3.0])
            with When("validating numerical finiteness"):
                with Then("a ValueError is raised pinpointing the presence of NaN"):
                    with pytest.raises(ValueError, match="NaN"):
                        validate_finite_tensor(nan_tensor, "corrupted")

        with Given("a corrupted tensor containing Inf values"):
            inf_tensor = torch.tensor([1.0, float("inf"), 3.0])
            with When("validating numerical finiteness"):
                with And("a ValueError is raised pinpointing the presence of Inf"):
                    with pytest.raises(ValueError, match="Inf"):
                        validate_finite_tensor(inf_tensor, "overflow")


def test_optional_dict_and_all_finite():
    with Scenario("Unit: Optional Dictionary and Batch Tensor Validation"):
        with Given("optional dictionary inputs"):
            with When("None is supplied"):
                empty_d = validate_optional_dict("config", None)
                with Then("an empty dictionary is returned"):
                    assert empty_d == {}

            with When("a valid dictionary is supplied"):
                d = validate_optional_dict("config", {"lr": 1e-4})
                with Then("the dictionary is returned intact"):
                    assert d == {"lr": 1e-4}

            with When("an invalid non-dict non-None type is supplied"):
                with Then("a TypeError is raised"):
                    with pytest.raises(TypeError):
                        validate_optional_dict("config", "not_a_dict")

        with Given("a sequence of named tensors"):
            t1 = torch.ones(2, 4)
            t2 = torch.zeros(4, 2)
            with When("validating all tensors in batch are finite"):
                validate_all_finite_tensors([("t1", t1), ("t2", t2)])
                with Then("all tensors pass cleanly"):
                    assert True
