import time
from collections.abc import Sequence
from dataclasses import dataclass

from constellation.core.controller import ControllerState, ScriptableController
from constellation.core.controller_configuration import load_config
from constellation.core.protocol.cscp1 import SatelliteState

from sparc_common import ChipParams, get_run_id_fhr, get_run_id_inj
import sparc_ireset_calculator, sparc_masking_generator


@dataclass
class InjectionSettings:
    n_injections: int
    pulse_interval: float
    cols: Sequence[int]
    rows: Sequence[int]


class Scanner:
    def __init__(
        self,
        config_file: str,
        group_name: str,
        data_dir: str,
        chip: str,
    ) -> None:
        self._ctrl = ScriptableController(group_name)
        self._cfg = load_config(config_file)
        self._output_directory_base = data_dir + "/" + chip

    def setup(self, data_subdir: str) -> None:
        # Wait until all satellites are connected
        self._ctrl.await_satellites(
            ["MessagePackWriter.SPARC", "Keithley.Bias", "Caribou.SPARC", "Influx.DB"]
        )

        # Land if in ORBIT
        if self._ctrl.state == ControllerState.ORBIT:
            self._ctrl.constellation.land()
            self._ctrl.await_state(SatelliteState.INIT, timeout=15)

        # Await INIT except if NEW
        initial_state = (
            SatelliteState.NEW
            if self._ctrl.state == ControllerState.NEW
            else SatelliteState.INIT
        )
        self._ctrl.await_state(initial_state, timeout=1)

        # Initialize
        self._ctrl.constellation.initialize(self._cfg)
        self._ctrl.await_state(SatelliteState.INIT, timeout=5)

        # Launch
        self._ctrl.constellation.launch()
        self._ctrl.await_state(SatelliteState.ORBIT, timeout=15)

        # Reconfigure output directory
        self._ctrl.constellation.MessagePackWriter.SPARC.reconfigure(
            {"output_directory": f"{self._output_directory_base}/{data_subdir}"}
        )
        self._ctrl.await_state(SatelliteState.ORBIT, timeout=5)

    def _reconfigure_sparc(self, cp: ChipParams) -> None:
        self._ctrl.log.status(
            "Reconfiguring"
            f"\n Vcasb to {cp.vcasb} mV"
            f"\n Ireset to {cp.ireset} pA"
            f"\n Vh to {cp.vh} mV"
            f"\n Vcasn to {cp.vcasn} mV"
            f"\n Ibias to {cp.ibias} nA"
            f"\n Idb to {cp.idb} nA"
            f"\n Vh to {cp.vh} mV"
        )
        recfg = {
            "devices": {
                "SPARC": {
                    "resources": {
                        "vcasb": cp.vcasb * 1e-3,
                        "vreset": sparc_ireset_calculator.ireset_pixel_to_vreset(
                            cp.ireset * 1e-12
                        ),
                        "vcasn": cp.vcasn * 1e-3,
                        "ibias": cp.ibias * 1e2 * 1e-9,
                        "ibiasn": cp.ibias * 1e2 * 1e-9,
                        "idb": cp.idb * 1e2 * 1e-9,
                        "vh": cp.vh * 1e-3,
                    }
                }
            }
        }
        self._ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
        self._ctrl.await_state(SatelliteState.ORBIT)

    def scan_iteration_inj(
        self,
        cp: ChipParams,
        injs: InjectionSettings,
    ) -> None:
        self._reconfigure_sparc(cp)

        for col in injs.cols:
            for row in injs.rows:
                # Reconfigure active pixel
                self._ctrl.log.status(f"Reconfiguring active pixel to ({col}, {row})")
                recfg = {
                    "devices": {
                        "SPARC": {
                            "registers": sparc_masking_generator.sparc_mask_all_pixels_except(
                                col, row
                            )
                        }
                    }
                }
                self._ctrl.constellation.Caribou.SPARC.reconfigure(recfg)
                self._ctrl.await_state(SatelliteState.ORBIT)

                # Start the run
                run_identifier = get_run_id_inj(cp, col, row)
                self._ctrl.log.status(f"Starting run {run_identifier}")
                self._ctrl.constellation.start(run_identifier)
                self._ctrl.await_state(SatelliteState.RUN)

                # Run injections
                for _ in range(injs.n_injections):
                    self._ctrl.constellation.Caribou.SPARC._set_memory(
                        ["sparc", "pulse_cmd", 1]
                    )
                    time.sleep(injs.pulse_interval)

                # Stop the run and await ORBIT state of all satellites
                self._ctrl.constellation.stop()
                self._ctrl.await_state(SatelliteState.ORBIT)

    def scan_iteration_fhr(
        self,
        cp: ChipParams,
        fhr_duration: float,
    ) -> None:
        self._reconfigure_sparc(cp)

        # Start run
        run_identifier = get_run_id_fhr(cp)
        self._ctrl.log.status(f"Starting run {run_identifier}")
        self._ctrl.constellation.start(run_identifier)
        self._ctrl.await_state(SatelliteState.RUN)

        # Wait for FHR duration
        time.sleep(fhr_duration)

        # Stop the run
        self._ctrl.constellation.stop()
        self._ctrl.await_state(SatelliteState.ORBIT)

    def land(self) -> None:
        self._ctrl.constellation.land()
        self._ctrl.await_state(SatelliteState.INIT, timeout=15)
