under Mesh {
  Type = ThreeDimensional;
  X0   = 0.0;
  Xmax = 1.0;
  Y0   = -0.05;
  Ymax = 0.05;
  Z0   = -0.05;
  Zmax = 0.05;
  NumberOfCellsX = 400;
  NumberOfCellsY = 1;
  NumberOfCellsZ = 1;

  BoundaryConditionX0   = Inlet;
  BoundaryConditionXmax = Inlet2;
  BoundaryConditionY0   = Symmetry;
  BoundaryConditionYmax = Symmetry;
  BoundaryConditionZ0   = Symmetry;
  BoundaryConditionZmax = Symmetry;
}

under Equations {
  under Material[0] {
    EquationOfState = StiffenedGas;
    under StiffenedGasModel {
      SpecificHeatRatio = 1.4;
      PressureConstant = 0.0;
    }
  }
  under Material[1] {
    EquationOfState = StiffenedGas;
    under StiffenedGasModel {
      SpecificHeatRatio = 1.4;
      PressureConstant = 0.0;
    }
  }
}

under InitialCondition {
  UserDefinedStateCalculator = "IC/libUserDefinedState.so";
}

under BoundaryConditions {
  under Inlet {
    MaterialID = 0;
    Density = 5.99924;
    VelocityX = 19.5975;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 460.894;
  }
  under Inlet2 {
    MaterialID = 0;
    Density = 5.99242;
    VelocityX = -6.19633;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 46.0950;
  }
}

under Space {
  under NavierStokes {
    Flux = LocalLaxFriedrichs;
    under Reconstruction {
      Type = Linear;
      VariableType = ConservativeCharacteristic; //not very sensitive
      SlopeNearInterface = NonZero;
      Limiter = VanAlbada; //GeneralizedMinMod;
      GeneralizedMinModCoefficient = 1.2;
    }
  }

  under LevelSet[0] {
    MaterialID = 1;
    under Reconstruction {
      Type = Linear;
      Limiter = None;
    }
  }

}

under MultiPhase {
  Flux = Numerical;
  ReconstructionAtInterface = Linear;  //slightly better than constant
  PhaseChange = RiemannSolution; //clearly better than extrapolation
}

under Time {
  Type = Explicit;
  MaxTime = 0.0245;
  TimeStep = 5e-6;
  under Explicit {
    Type = RungeKutta3;
  }
}

under Output {
  Prefix = "results/";
  Solution = "solution";
  TimeInterval = 0.00035;
  Density = On;
  Velocity = On;
  Pressure = On;
  LevelSet0 = On;
  MaterialID = On;
  Temperature = On;
}
